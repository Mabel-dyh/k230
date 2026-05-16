# train.py — user CSV training (uses shared model.py)
import os
import uuid
import pickle

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from csv_utils import normalize_training_csv
from model import (
    FastDelayPredictLSTM,
    HIDDEN_DIM,
    INPUT_DIM,
    NUM_LAYERS,
    DROPOUT,
    LOSS_ALPHA,
    LOSS_BETA,
    RANDOM_SEED,
    loss_function,
)

MODEL_DIR = "user_models"
UPLOAD_DIR = "uploads"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _stage_positions(length):
    if length > 1:
        return np.arange(length, dtype=np.float32) / (length - 1)
    return np.array([0.0], dtype=np.float32)


def _compute_feature_stats(padded_data, actual_stages):
    """Mean/std over valid (non-pad) positions only."""
    mask = np.zeros(padded_data.shape[:2], dtype=bool)
    for i, L in enumerate(actual_stages):
        mask[i, :L] = True
    valid = padded_data[mask]
    mean = valid.mean(axis=0).astype(np.float32)
    std = valid.std(axis=0).astype(np.float32) + 1e-9
    return mean, std


class RealInverterChainDataset(Dataset):
    """Long format (chain_id + stage_idx) or legacy per-row format."""

    def __init__(self, df):
        if df.empty:
            raise ValueError("CSV is empty")

        self.data = []
        self.targets_stage = []
        self.targets_total = []
        self.actual_stages = []

        if "chain_id" in df.columns and "stage_idx" in df.columns:
            for _, group in df.groupby("chain_id"):
                g = group.sort_values("stage_idx")
                stages = len(g)
                features = g[["W_um", "CL_fF", "VDD_V"]].values.astype(np.float32)

                if "stage_delay_ps" in g.columns and not g["stage_delay_ps"].isna().all():
                    stage_delay = g["stage_delay_ps"].values.astype(np.float32)
                else:
                    total_delay = float(
                        g["total_delay_ps"].iloc[0]
                        if "total_delay_ps" in g.columns
                        else g["delay_ps"].iloc[0]
                    )
                    stage_delay = np.full(stages, total_delay / stages, dtype=np.float32)

                total_delay = float(
                    g["total_delay_ps"].iloc[0]
                    if "total_delay_ps" in g.columns
                    else g["delay_ps"].iloc[0]
                )

                self.data.append(features)
                self.targets_stage.append(stage_delay)
                self.targets_total.append(total_delay)
                self.actual_stages.append(stages)
        else:
            for _, row in df.iterrows():
                w, cl, vdd = float(row["W_um"]), float(row["CL_fF"]), float(row["VDD_V"])
                stages = int(row["stages"])
                total_delay = float(row["delay_ps"])
                features = np.tile([w, cl, vdd], (stages, 1)).astype(np.float32)
                stage_delay = np.full(stages, total_delay / stages, dtype=np.float32)
                self.data.append(features)
                self.targets_stage.append(stage_delay)
                self.targets_total.append(total_delay)
                self.actual_stages.append(stages)

        max_seq = max(self.actual_stages)
        padded_data = []
        padded_targets_stage = []
        for i in range(len(self.data)):
            feat = self.data[i]
            L = feat.shape[0]
            stage_pos = _stage_positions(L)
            feat_with_pos = np.concatenate([feat, stage_pos.reshape(-1, 1)], axis=1)
            pad_len = max_seq - L
            padded_data.append(
                np.pad(feat_with_pos, ((0, pad_len), (0, 0)), constant_values=0.0)
            )
            padded_targets_stage.append(
                np.pad(self.targets_stage[i], (0, pad_len), constant_values=0.0)
            )

        self.data = np.array(padded_data, dtype=np.float32)
        self.targets_stage = np.array(padded_targets_stage, dtype=np.float32)
        self.targets_total = np.array(self.targets_total, dtype=np.float32)
        self.actual_stages = np.array(self.actual_stages, dtype=np.int64)
        self.seq_len = max_seq

        self.feature_mean, self.feature_std = _compute_feature_stats(self.data, self.actual_stages)
        self.data = (self.data - self.feature_mean) / self.feature_std
        self.targets_stage_log = np.log(self.targets_stage + 1e-9)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.targets_stage_log[idx], dtype=torch.float32),
            torch.tensor(self.targets_total[idx], dtype=torch.float32),
            torch.tensor(self.actual_stages[idx], dtype=torch.long),
        )


def train_from_csv_lstm(csv_path, model_name=""):
    """
    Generator: yields progress text; final yield is
    (text, model_id, target_col, pth_path, scalers_path).
    """
    model_id = model_name.strip() if model_name.strip() else str(uuid.uuid4())
    messages = []

    def report(msg):
        messages.append(msg)
        return "\n".join(messages)

    yield report("📄 加载 CSV...")

    df = normalize_training_csv(pd.read_csv(csv_path))
    if df.empty:
        raise ValueError("CSV 为空")
    target_col = "total_delay_ps" if "total_delay_ps" in df.columns else "delay_ps"

    dataset = RealInverterChainDataset(df)
    yield report(f"📊 数据: {len(dataset)} chains, max stages: {dataset.seq_len}")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if len(dataset) > 1:
        train_size = max(1, int(0.8 * len(dataset)))
        val_size = len(dataset) - train_size
        train_ds, val_ds = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(RANDOM_SEED),
        )
    else:
        train_ds = val_ds = dataset

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastDelayPredictLSTM(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 50
    patience = 5
    best_val_loss = float("inf")
    no_improve = 0
    best_model_state = model.state_dict()

    for epoch in range(num_epochs):
        yield report(f"🔄 Epoch {epoch + 1}/{num_epochs}...")
        model.train()
        for batch in train_loader:
            x_batch, y_stage_log_batch, y_total_batch, stages_batch = batch
            x_batch = x_batch.to(device)
            y_stage_log_batch = y_stage_log_batch.to(device)
            y_total_batch = y_total_batch.to(device)
            stages_batch = stages_batch.to(device)

            optimizer.zero_grad()
            mu, logvar = model(x_batch, lengths=stages_batch)
            loss, _, _ = loss_function(
                mu, logvar, y_stage_log_batch, y_total_batch, stages_batch, LOSS_ALPHA, LOSS_BETA
            )
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_batch, y_stage_log_batch, y_total_batch, stages_batch = batch
                x_batch = x_batch.to(device)
                y_stage_log_batch = y_stage_log_batch.to(device)
                y_total_batch = y_total_batch.to(device)
                stages_batch = stages_batch.to(device)
                mu, logvar = model(x_batch, lengths=stages_batch)
                loss, _, _ = loss_function(
                    mu, logvar, y_stage_log_batch, y_total_batch, stages_batch, LOSS_ALPHA, LOSS_BETA
                )
                val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        if avg_val_loss < best_val_loss - 1e-12:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            yield report(f"✅ 新的最好模型 (val loss={best_val_loss:.6e})")
        else:
            no_improve += 1
            yield report(f"⏳ 无提升计数: {no_improve}/{patience}")
            if no_improve >= patience:
                yield report("⏹️ 早停训练")
                break

    yield report("💾 保存模型与 scalers...")

    scalers = {"feature_mean": dataset.feature_mean, "feature_std": dataset.feature_std}
    scalers_path = os.path.join(MODEL_DIR, f"{model_id}_scalers.pkl")
    pth_path = os.path.join(MODEL_DIR, f"{model_id}.pth")
    with open(scalers_path, "wb") as f:
        pickle.dump(scalers, f)
    torch.save(best_model_state, pth_path)

    yield report(f"✅ 模型训练完成: {model_id}, 目标列: {target_col}")
    yield ("", model_id, target_col, pth_path, scalers_path)
