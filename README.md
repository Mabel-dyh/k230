# 晶体管反相器链延迟预测

基于 LSTM 的 per-stage 延迟预测，Gradio Web UI + 用户 CSV 在线训练。

## 运行

```bash
pip install -r requirements.txt
python app.py
```

预训练默认模型（可选，会覆盖根目录 `inverter_model_state.pth` / `scalers.pkl`）：

```bash
python learn.py
```

命令行预测：

```bash
python predict.py --W 2.0,2.0,2.0 --CL 5.0,5.0,5.0 --VDD 1.0,1.0,1.0
```

## CSV 格式

**长格式（推荐）**：`chain_id`, `stage_idx`, `W_um`, `CL_fF`, `VDD_V`, `stage_delay_ps`（可选）, `total_delay_ps`

**旧格式**：每行一条链，`W_um`, `CL_fF`, `VDD_V`, `stages`, `delay_ps`

列名别名：`Weight_um` → `W_um` 等（见 `csv_utils.py`）

## 项目结构

| 文件 | 说明 |
|------|------|
| `app.py` | Gradio 界面 |
| `model.py` | 共享 LSTM 模型与损失 |
| `inference.py` | 预测与模型缓存 |
| `train.py` | 用户 CSV 训练 |
| `learn.py` | 合成数据预训练 |
| `predict.py` | CLI 预测 |

用户模型保存在 `user_models/`。
