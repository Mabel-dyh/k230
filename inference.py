"""Model loading (cached) and chain delay prediction."""
import os
import pickle
from functools import lru_cache

import numpy as np
import torch

from model import FastDelayPredictLSTM, HIDDEN_DIM, INPUT_DIM, NUM_LAYERS, DROPOUT

DEFAULT_MODEL_PTH = "inverter_model_state.pth"
DEFAULT_SCALERS_PKL = "scalers.pkl"


def load_scalers(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _load_state_dict(model, ckpt_path):
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt)


@lru_cache(maxsize=8)
def _cached_model_bundle(cache_key):
    """cache_key: (abs_pth, abs_pkl, mtime_pth, mtime_pkl)"""
    pth_path, pkl_path = cache_key[0], cache_key[1]
    scalers = load_scalers(pkl_path)
    model = FastDelayPredictLSTM(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT
    )
    _load_state_dict(model, pth_path)
    model.eval()
    mean = np.array(scalers["feature_mean"], dtype=np.float32)
    std = np.array(scalers["feature_std"], dtype=np.float32)
    return model, mean, std


def get_model_and_scalers(model_pth, scalers_pkl):
    pth_path = os.path.abspath(model_pth)
    pkl_path = os.path.abspath(scalers_pkl)
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"模型文件未找到: {pth_path}")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Scalers 文件未找到: {pkl_path}")
    mtime_pth = os.path.getmtime(pth_path)
    mtime_pkl = os.path.getmtime(pkl_path)
    return _cached_model_bundle((pth_path, pkl_path, mtime_pth, mtime_pkl))


def clear_model_cache():
    _cached_model_bundle.cache_clear()


def parse_stage_lists(w_str, cl_str, vdd_str):
    w_list = [float(w.strip()) for w in w_str.split(",") if w.strip()]
    cl_list = [float(c.strip()) for c in cl_str.split(",") if c.strip()]
    vdd_list = [float(v.strip()) for v in vdd_str.split(",") if v.strip()]
    n = len(w_list)
    if n == 0 or len(cl_list) != n or len(vdd_list) != n:
        raise ValueError("输入格式错误：W, CL, VDD 三个列表须长度一致且非空（逗号分隔）")
    return w_list, cl_list, vdd_list


def predict_chain(w_list, cl_list, vdd_list, model_pth=DEFAULT_MODEL_PTH, scalers_pkl=DEFAULT_SCALERS_PKL):
    """Run forward pass; returns dict of arrays and scalars."""
    model, mean, std = get_model_and_scalers(model_pth, scalers_pkl)
    n = len(w_list)
    if mean.shape[0] != 4 or std.shape[0] != 4:
        raise ValueError(
            f"scalers 期待 4 维特征，得到 mean.shape={mean.shape}, std.shape={std.shape}"
        )

    if n > 1:
        stage_pos = np.arange(n, dtype=np.float32) / (n - 1)
    else:
        stage_pos = np.array([0.0], dtype=np.float32)

    x_arr = np.stack(
        [
            np.array(w_list, dtype=np.float32),
            np.array(cl_list, dtype=np.float32),
            np.array(vdd_list, dtype=np.float32),
        ],
        axis=1,
    )
    x_full = np.concatenate([x_arr, stage_pos.reshape(-1, 1)], axis=1)
    x_std = (x_full - mean) / (std + 1e-9)
    x_tensor = torch.tensor(x_std.reshape(1, n, 4), dtype=torch.float32)

    with torch.no_grad():
        lengths = torch.tensor([n], dtype=torch.long)
        mu, logvar = model(x_tensor, lengths=lengths)
        mu = mu.cpu().numpy().reshape(-1)
        logvar = logvar.cpu().numpy().reshape(-1)

    var = np.exp(logvar)
    pred_stage = np.exp(mu + 0.5 * var)
    pred_total = float(pred_stage.sum())

    base_delays = np.array(cl_list, dtype=np.float64) / (
        np.array(w_list, dtype=np.float64) * np.array(vdd_list, dtype=np.float64)
    )
    theoretical_total = float(base_delays.sum())

    rng = np.random.default_rng(0)
    stage_samps = rng.normal(loc=mu, scale=np.sqrt(var), size=(2000, mu.size))
    total_samps = np.exp(stage_samps).sum(axis=1)
    p05, p50, p95 = np.percentile(total_samps, [5, 50, 95])

    return {
        "n": n,
        "w_list": w_list,
        "cl_list": cl_list,
        "vdd_list": vdd_list,
        "pred_stage": pred_stage,
        "pred_total": pred_total,
        "base_delays": base_delays,
        "theoretical_total": theoretical_total,
        "p05": p05,
        "p50": p50,
        "p95": p95,
    }


def format_prediction_report(result, k=1e-12):
    w_list = result["w_list"]
    cl_list = result["cl_list"]
    vdd_list = result["vdd_list"]
    n = result["n"]
    lines = ["=== 输入 (per-stage) ==="]
    for i in range(n):
        lines.append(f"Stage {i+1}: W={w_list[i]} μm, CL={cl_list[i]} fF, VDD={vdd_list[i]} V")
    lines.append(f"Total stages: {n}")
    lines.append("\n=== 理论延迟（简化） ===")
    lines.append(
        f"Per-stage base_delay: {np.array2string(result['base_delays'], precision=6, separator=', ')}"
    )
    lines.append(f"理论总延迟 = {result['theoretical_total']:.6e}")
    lines.append("\n=== 模型预测（对数正态均值） ===")
    lines.append(
        f"Per-stage predicted mean: {np.array2string(result['pred_stage'], precision=6, separator=', ')}"
    )
    lines.append(f"总延迟 point estimate = {result['pred_total']:.6e}")
    lines.append(
        f"sample median = {result['p50']:.6e}, 5%-95% = [{result['p05']:.6e}, {result['p95']:.6e}]"
    )
    rel_err = (
        abs(result["pred_total"] - result["theoretical_total"])
        / (result["theoretical_total"] + 1e-15)
        * 100.0
    )
    lines.append(f"相对误差 (vs 理论) = {rel_err:.3f}%")
    lines.append(f"\n=== 换算为秒 (k={k:.1e}) ===")
    lines.append(
        f"预测平均 = {result['pred_total'] * k:.6e} s ; 理论 = {result['theoretical_total'] * k:.6e} s"
    )
    return "\n".join(lines)
