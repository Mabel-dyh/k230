import os
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle

# 保持与原 learn.py 一致的可视化配置（不影响导出）
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.autolayout'] = True

# -----------------------
# 复刻随机种子（保证可复现）
# -----------------------
np.random.seed(0)
torch.manual_seed(0)

# -----------------------
# 复刻 InverterChainDataset（与 learn.py 完全一致）
# -----------------------
class InverterChainDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=5):
        self.num_samples = num_samples
        self.seq_len = seq_len
        # 扩展W, CL, VDD的范围并增加噪声水平
        # W: [0.5, 5.0]; CL: [0.1, 10.0]; VDD: [0.6, 1.2]
        self.data = []
        self.targets_stage = []
        self.targets_total = []
        for _ in range(num_samples):
            W = np.random.lognormal(mean=0, sigma=0.5, size=seq_len).astype(np.float32)
            CL = np.random.lognormal(mean=0, sigma=0.5, size=seq_len).astype(np.float32)
            VDD = np.random.normal(loc=0.9, scale=0.2, size=seq_len).astype(np.float32)
            W = np.clip(W, 0.5, 5.0)  # 限制范围
            CL = np.clip(CL, 0.1, 10.0)
            VDD = np.clip(VDD, 0.6, 1.2)
            # 模拟每级反相器延迟 (CL / (W * VDD)) 加上噪声
            base_delay = CL / (W * VDD)
            noise = np.random.normal(0, 0.1 * base_delay)  # 噪声比例0.1
            stage_delay = base_delay + noise
            stage_delay = np.clip(stage_delay, a_min=0.0, a_max=None)  # 保证非负
            total_delay = np.sum(stage_delay)
            features = np.stack([W, CL, VDD], axis=1)  # (seq_len, 3)
            self.data.append(features)
            self.targets_stage.append(stage_delay)
            self.targets_total.append(total_delay)
        self.data = np.array(self.data)  # (num_samples, seq_len, 3)
        self.targets_stage = np.array(self.targets_stage)  # (num_samples, seq_len)
        self.targets_total = np.array(self.targets_total)  # (num_samples,)

        # 特征归一化 (按列)
        self.feature_mean = self.data.mean(axis=(0,1))
        self.feature_std = self.data.std(axis=(0,1))
        self.data = (self.data - self.feature_mean) / (self.feature_std + 1e-6)

        # 对延迟取对数 (log尺度训练)
        self.targets_stage_log = np.log(self.targets_stage + 1e-6)
        self.targets_total_log = np.log(self.targets_total + 1e-6)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)        # 特征（归一化后）
        y_stage_log = torch.tensor(self.targets_stage_log[idx], dtype=torch.float32)
        y_total_log = torch.tensor(self.targets_total_log[idx], dtype=torch.float32)
        return x, y_stage_log, y_total_log

# -----------------------
# 主流程：生成 dataset 并导出 CSV + scalers.pkl
# -----------------------
def export_dataset_to_csv(dataset, save_dir="./generated_data"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(save_dir, f"default_dataset_{timestamp}.csv")
    rows = []

    # 注意：dataset.data 已被归一化，因此导出时需要还原：
    # 原始_features = normalized * feature_std + feature_mean
    for i in range(len(dataset)):
        features = dataset.data[i] * dataset.feature_std + dataset.feature_mean  # (seq_len, 3)
        stage_delays = dataset.targets_stage[i]   # 线性尺度
        total_delay = dataset.targets_total[i]    # 线性尺度
        for s in range(dataset.seq_len):
            rows.append({
                "sample_id": int(i),
                "stage_idx": int(s),
                "W": float(features[s, 0]),
                "CL": float(features[s, 1]),
                "VDD": float(features[s, 2]),
                "stage_delay": float(stage_delays[s]),
                "total_delay": float(total_delay),
            })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[OK] 导出 CSV: {csv_path}")

    # 同时保存 scalers.pkl（与 learn.py 保存格式一致）
    scalers = {
        "feature_mean": dataset.feature_mean,
        "feature_std": dataset.feature_std
    }
    pkl_path = os.path.join(save_dir, f"scalers_{timestamp}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(scalers, f)
    print(f"[OK] 保存 scalers: {pkl_path}")

    return csv_path, pkl_path

if __name__ == "__main__":
    # 复刻 learn.py 中的默认参数（num_samples=5000, seq_len=5）
    dataset = InverterChainDataset(num_samples=5000, seq_len=5)

    # 导出 CSV（每级一行）
    export_dataset_to_csv(dataset, save_dir="./generated_data")
    print("完成。")