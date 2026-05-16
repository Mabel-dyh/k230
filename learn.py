# learn.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from model import (
    FastDelayPredictLSTM,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT,
    LOSS_ALPHA,
    LOSS_BETA,
    loss_function,
)
import matplotlib.pyplot as plt
import tqdm
import pickle
import os

# ---------------------------
# 配置（可按需调整）
# ---------------------------
NUM_SAMPLES = 20000
MIN_STAGES = 3
MAX_STAGES = 15
CSV_FILENAME = "inverter_chain_dataset.csv"
SCALERS_PKL = "scalers.pkl"
MODEL_PTH = "inverter_model_state.pth"

BATCH_SIZE = 64
NUM_EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
CLIP_NORM = 2.0

PATIENCE = 8  # early stopping patience
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------
# 数据生成
# ---------------------------
def generate_inverter_data(num_samples=NUM_SAMPLES, min_stages=MIN_STAGES, max_stages=MAX_STAGES, filename=CSV_FILENAME):
    """
    生成逆变器链数据，每行对应一个 stage；新增 stage_pos 特征（stage index normalized）。
    Delay 建模引入了后级耦合（next stage CL 的一部分）来模拟实际耦合。
    """
    print(f"--- Generating data -> {filename} (samples={num_samples}) ---")
    records = []
    for sample_id in tqdm.tqdm(range(num_samples), desc="Generating Samples"):
        num_stages = np.random.randint(min_stages, max_stages + 1)
        # 基本参数：使用 lognormal / normal，并 clip 措施
        W = np.random.lognormal(mean=0.5, sigma=1.2, size=num_stages).astype(np.float32)
        CL = np.random.lognormal(mean=1.0, sigma=1.5, size=num_stages).astype(np.float32)
        VDD = np.random.normal(loc=1.0, scale=0.1, size=num_stages).astype(np.float32)

        W = np.clip(W, 0.5, 15.0)
        CL = np.clip(CL, 0.1, 25.0)
        VDD = np.clip(VDD, 0.8, 1.2)

        # 设计耦合：每级 delay 受后级 CL 的影响（factor）
        coupling_factor = np.random.uniform(0.2, 0.6)  # 随样本略有差异
        base_delay = np.zeros(num_stages, dtype=np.float32)
        for i in range(num_stages):
            next_cl = CL[i+1] if i+1 < num_stages else 0.0
            base_delay[i] = (CL[i] + coupling_factor * next_cl) / (W[i] * VDD[i])

        # 加入噪声（相对 base_delay）
        noise = np.random.normal(0.0, 0.05 * base_delay)
        stage_delay = np.clip(base_delay + noise, a_min=1e-9, a_max=None).astype(np.float32)
        total_delay = np.sum(stage_delay).astype(np.float32)

        for stage_id in range(num_stages):
            stage_pos = stage_id / (num_stages - 1) if num_stages > 1 else 0.0
            records.append({
                'sample_id': sample_id,
                'Weight_um': float(W[stage_id]),
                'CL_fF': float(CL[stage_id]),
                'VDD_V': float(VDD[stage_id]),
                'stage_pos': float(stage_pos),
                'stage_delay': float(stage_delay[stage_id]),
                'num_stages': num_stages,
                'sample_total_delay': float(total_delay)
            })

    df = pd.DataFrame(records)
    df.to_csv(filename, index=False, float_format='%.8f')
    print(f"Data generation complete. Total rows: {len(df)}, saved to {filename}")
    return df, filename

# ---------------------------
# Dataset & Collate
# ---------------------------
class InverterCSVDataset(Dataset):
    """
    自定义 Dataset：以 sample_id 分组，每个样本返回 (features_seq, y_stage_log)
    features order: ['Weight_um', 'CL_fF', 'VDD_V', 'stage_pos'] -> input_dim=4
    """
    def __init__(self, csv_file, feature_mean, feature_std):
        self.mean = torch.tensor(feature_mean, dtype=torch.float32)
        self.std = torch.tensor(feature_std, dtype=torch.float32)
        df = pd.read_csv(csv_file)
        # groupby sample_id (保持 list 形式)
        grouped = df.groupby('sample_id')
        # store each grouped DataFrame as values to avoid repeated disk I/O
        self.samples = []
        for sid, g in grouped:
            # sort by stage position just in case
            g_sorted = g.sort_values(by='stage_pos')
            self.samples.append((sid, g_sorted.reset_index(drop=True)))
        # keep column names for debugging:
        self.feature_cols = ['Weight_um', 'CL_fF', 'VDD_V', 'stage_pos']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id, sample_df = self.samples[idx]
        features = sample_df[['Weight_um', 'CL_fF', 'VDD_V', 'stage_pos']].values.astype(np.float32)
        stage_delays = sample_df['stage_delay'].values.astype(np.float32)
        # 标准化（使用传入的 mean/std）
        x = (torch.tensor(features, dtype=torch.float32) - self.mean) / (self.std + 1e-9)
        y_stage_log = torch.log(torch.tensor(stage_delays, dtype=torch.float32) + 1e-9)
        return x, y_stage_log

def pad_collate_fn(batch):
    """
    batch: list of (seq_tensor (L, D), target_tensor (L,))
    返回: padded_sequences (B, Lmax, D), padded_targets (B, Lmax), lengths (B,)
    - 按长度降序排序，以便 pack_padded_sequence 使用
    """
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    sequences, targets = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long)
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
    padded_targets = nn.utils.rnn.pad_sequence([t.unsqueeze(1) for t in targets], batch_first=True, padding_value=0.0).squeeze(-1)
    return padded_sequences, padded_targets, lengths

def _y_total_from_stage_log(y_stage_log, lengths):
    device = y_stage_log.device
    max_len = y_stage_log.size(1)
    mask = torch.arange(max_len, device=device)[None, :] < lengths[:, None]
    true_stage = torch.exp(y_stage_log)
    return torch.sum(true_stage * mask.float(), dim=1)


def compute_metrics(y_true_total, y_pred_total):
    # arrays
    y_true = np.array(y_true_total)
    y_pred = np.array(y_pred_total)
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    rel = np.abs((y_pred - y_true) / (y_true + 1e-9)) * 100.0
    return {'MAE': mae, 'RMSE': rmse, 'mean_rel_%': np.mean(rel), 'median_rel_%': np.median(rel), '90pct_rel_%': np.percentile(rel, 90)}

# ---------------------------
# 主流程
# ---------------------------
def main():
    # 1) 生成数据（如果文件存在可跳过；但按要求我们重新生成覆盖）
    df, csv_filename = generate_inverter_data()

    # 2) 计算 normalization stats（包含 stage_pos）
    feature_cols = ['Weight_um', 'CL_fF', 'VDD_V', 'stage_pos']
    feature_mean = df[feature_cols].mean().values.astype(np.float32)
    feature_std = df[feature_cols].std().values.astype(np.float32)
    print("Feature mean:", feature_mean)
    print("Feature std :", feature_std)
    scalers = {"feature_mean": feature_mean, "feature_std": feature_std}
    with open(SCALERS_PKL, "wb") as f:
        pickle.dump(scalers, f)
    print(f"Saved scalers to {SCALERS_PKL} (will overwrite if exists)")

    # 3) 创建 Dataset / DataLoader
    full_dataset = InverterCSVDataset(csv_filename, feature_mean, feature_std)
    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)
    print(f"Dataset sizes - total: {n_total}, train: {len(train_ds)}, val: {len(val_ds)}")

    # 4) model / optimizer / scheduler
    model = FastDelayPredictLSTM(input_dim=len(feature_cols), hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # 方案 B：不传 verbose（兼容旧版 PyTorch），我们会手动打印 lr 变化
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val = float('inf')
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    # 保存当前 lr 以便之后比较
    last_lr = optimizer.param_groups[0]['lr']

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch, lengths in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            lengths = lengths.to(DEVICE)
            optimizer.zero_grad()
            mu, logvar = model(x_batch, lengths=lengths)
            y_total = _y_total_from_stage_log(y_batch, lengths)
            loss, stage_nll_val, mse_total_val = loss_function(
                mu, logvar, y_batch, y_total, lengths, LOSS_ALPHA, LOSS_BETA
            )
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch, lengths in val_loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                lengths = lengths.to(DEVICE)
                mu, logvar = model(x_batch, lengths=lengths)
                y_total = _y_total_from_stage_log(y_batch, lengths)
                loss, _, _ = loss_function(
                    mu, logvar, y_batch, y_total, lengths, LOSS_ALPHA, LOSS_BETA
                )
                running_val_loss += loss.item()
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.6e} | Val Loss: {avg_val_loss:.6e}")

        # scheduler step (使用 val loss)
        scheduler.step(avg_val_loss)

        # 手动检测 lr 是否变化并打印（兼容所有 PyTorch 版本）
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != last_lr:
            print(f"Learning rate reduced: {last_lr:.3e} -> {current_lr:.3e}")
        last_lr = current_lr

        # checkpoint
        if avg_val_loss < best_val - 1e-12:
            best_val = avg_val_loss
            epochs_no_improve = 0
            # save state_dict only
            torch.save(model.state_dict(), MODEL_PTH)
            print(f"  -> New best model saved to {MODEL_PTH}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered (no improvement for {PATIENCE} epochs).")
                break

    # 5) 评估并绘图
    # reload best model
    if os.path.exists(MODEL_PTH):
        model.load_state_dict(torch.load(MODEL_PTH, map_location=DEVICE))
        print(f"Loaded best model from {MODEL_PTH} for evaluation.")
    else:
        print("Warning: Best model file not found; evaluating current model.")

    # plot losses
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    print("Saved loss_curve.png")

    # compute predictions on validation set (total delay)
    model.eval()
    y_true_total, y_pred_total = [], []
    with torch.no_grad():
        for x_batch, y_batch, lengths in val_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            lengths = lengths.to(DEVICE)
            mu, logvar = model(x_batch, lengths=lengths)
            # clamp logvar and compute var
            logvar = torch.clamp(logvar, min=-10.0, max=10.0)
            var = torch.exp(logvar)
            pred_stage_linear = torch.exp(mu + 0.5 * var)
            true_stage_linear = torch.exp(y_batch)
            max_len = mu.size(1)
            mask = torch.arange(max_len, device=DEVICE)[None, :] < lengths[:, None]
            pred_total = torch.sum(torch.where(mask, pred_stage_linear, torch.zeros_like(pred_stage_linear)), dim=1)
            true_total = torch.sum(torch.where(mask, true_stage_linear, torch.zeros_like(true_stage_linear)), dim=1)
            y_pred_total.extend(pred_total.cpu().numpy().tolist())
            y_true_total.extend(true_total.cpu().numpy().tolist())

    metrics = compute_metrics(y_true_total, y_pred_total)
    print("Evaluation metrics on validation set:")
    for k, v in metrics.items():
        if 'rel' in k:
            print(f"  {k}: {v:.2f}%")
        else:
            print(f"  {k}: {v:.6f}")

    # scatter plot predicted vs true
    plt.figure(figsize=(6,6))
    plt.scatter(y_true_total, y_pred_total, alpha=0.3, s=8)
    mmin = min(min(y_true_total), min(y_pred_total))
    mmax = max(max(y_true_total), max(y_pred_total))
    plt.plot([mmin, mmax], [mmin, mmax], 'r--', label='ideal')
    plt.xlabel("True total delay")
    plt.ylabel("Predicted total delay")
    plt.title("Predicted vs True total delay (val set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pred_vs_true.png")
    print("Saved pred_vs_true.png")

    # already saved scalers at start; ensure overwritten with latest (redundant but safe)
    with open(SCALERS_PKL, "wb") as f:
        pickle.dump({"feature_mean": feature_mean, "feature_std": feature_std}, f)
    print(f"Final scalers saved to {SCALERS_PKL}")

if __name__ == "__main__":
    main()
