# app.py — Gradio UI
import os

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import pandas as pd

from inference import (
    DEFAULT_MODEL_PTH,
    DEFAULT_SCALERS_PKL,
    clear_model_cache,
    format_prediction_report,
    parse_stage_lists,
    predict_chain,
)
from train import MODEL_DIR, UPLOAD_DIR, train_from_csv_lstm

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Theme & CSS
# ---------------------------------------------------------------------------
APP_THEME = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="linear-gradient(160deg, #f0f4f8 0%, #e8eef5 45%, #f5f7fa 100%)",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_label_text_weight="600",
    button_primary_background_fill="linear-gradient(135deg, #0891b2 0%, #0e7490 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #0e7490 0%, #155e75 100%)",
    button_primary_text_color="#ffffff",
    input_background_fill="#f8fafc",
)

CUSTOM_CSS = """
#app-header {
    text-align: center;
    padding: 1.75rem 1.5rem 1.25rem;
    margin-bottom: 0.5rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 55%, #0e7490 100%);
    color: #f8fafc;
    box-shadow: 0 8px 32px rgba(15, 23, 42, 0.18);
}
#app-header h1 {
    margin: 0 0 0.35rem 0;
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #ffffff !important;
}
#app-header p {
    margin: 0;
    opacity: 0.88;
    font-size: 0.95rem;
    color: #e2e8f0 !important;
}
.model-bar {
    padding: 0.85rem 1rem !important;
    border-radius: 12px !important;
    border: 1px solid #e2e8f0 !important;
    background: #ffffff !important;
}
.panel-card {
    padding: 1rem !important;
    border-radius: 12px !important;
    border: 1px solid #e2e8f0 !important;
    background: #ffffff !important;
    height: 100%;
}
#train-output textarea {
    font-family: ui-monospace, "Cascadia Code", "Consolas", monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.55 !important;
}
.run-btn button {
    min-height: 2.75rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}
.tag-row span {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    margin: 0.15rem 0.25rem 0.15rem 0;
    border-radius: 999px;
    background: #ecfeff;
    color: #0e7490;
    font-size: 0.78rem;
    font-weight: 500;
    border: 1px solid #a5f3fc;
}
footer { display: none !important; }
"""


def list_available_models():
    names = ["default"]
    if os.path.isdir(MODEL_DIR):
        for fname in os.listdir(MODEL_DIR):
            if fname.endswith(".pth"):
                names.append(os.path.splitext(fname)[0])
    return sorted(list(dict.fromkeys(names)))


def refresh_model_list():
    choices = list_available_models()
    return gr.update(choices=choices, value=choices[0]), choices


def report_to_markdown(text: str) -> str:
    if text.startswith("❌"):
        return f"```\n{text}\n```"
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("===") and stripped.endswith("==="):
            title = stripped.strip("= ").strip()
            lines.append(f"\n### {title}\n")
        elif stripped:
            lines.append(line)
        else:
            lines.append("")
    return "\n".join(lines).strip()


def _empty_plot(message: str):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, color="#64748b")
    fig.tight_layout()
    return fig


def build_stage_chart(result: dict):
    """Per-stage bar chart: model prediction vs simplified theory."""
    n = result["n"]
    labels = [f"S{i + 1}" for i in range(n)]
    x = range(n)
    width = 0.38

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        [i - width / 2 for i in x],
        result["pred_stage"],
        width,
        label="模型预测",
        color="#0891b2",
        alpha=0.88,
    )
    ax.bar(
        [i + width / 2 for i in x],
        result["base_delays"],
        width,
        label="理论估算",
        color="#94a3b8",
        alpha=0.85,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Delay (relative units)")
    ax.set_title("Per-stage Delay")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig


def build_total_ci_chart(result: dict):
    """Total delay point estimate with 5–95% interval and theory reference."""
    p05, p50, p95 = result["p05"], result["p50"], result["p95"]
    pred_total = result["pred_total"]
    theoretical = result["theoretical_total"]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    y = 0.5
    ax.hlines(y, p05, p95, colors="#0891b2", linewidth=10, alpha=0.35, label="5%–95% 区间")
    ax.plot([p05, p95], [y, y], "|", color="#0e7490", markersize=14, markeredgewidth=2)
    ax.plot(p50, y, "o", color="#0e7490", markersize=9, label=f"采样中位数 ({p50:.4g})")
    ax.plot(
        pred_total,
        y,
        "s",
        color="#f97316",
        markersize=8,
        label=f"点估计 ({pred_total:.4g})",
    )
    ax.plot(
        theoretical,
        y,
        "x",
        color="#64748b",
        markersize=10,
        markeredgewidth=2,
        label=f"理论值 ({theoretical:.4g})",
    )
    ax.set_yticks([])
    ax.set_xlabel("Total Delay (relative units)")
    ax.set_title("Total Delay Uncertainty")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig


def run_predict(selected_model, w_str, cl_str, vdd_str, k):
    empty = _empty_plot("运行预测后显示图表")
    try:
        w_list, cl_list, vdd_list = parse_stage_lists(w_str, cl_str, vdd_str)
        if selected_model == "default":
            pth, pkl = DEFAULT_MODEL_PTH, DEFAULT_SCALERS_PKL
        else:
            pth = os.path.join(MODEL_DIR, f"{selected_model}.pth")
            pkl = os.path.join(MODEL_DIR, f"{selected_model}_scalers.pkl")
        result = predict_chain(w_list, cl_list, vdd_list, pth, pkl)
        md = report_to_markdown(format_prediction_report(result, k=float(k)))
        return md, build_stage_chart(result), build_total_ci_chart(result)
    except Exception as e:
        err = str(e)
        if "shape" in err.lower() or "size mismatch" in err.lower() or "Missing key" in err:
            md = report_to_markdown(
                f"❌ 无法加载模型（参数形状不匹配或文件损坏）:\n{err}\n\n"
                "请用当前训练代码重新训练模型或选择其他模型。"
            )
        else:
            md = report_to_markdown(f"❌ 出错: {err}")
        err_fig = _empty_plot("预测失败，无法绘图")
        return md, err_fig, err_fig


def train_gen(csv, name, models):
    if csv is None:
        yield "❌ 请上传 CSV 文件", models, None, None
        return

    try:
        last_text = ""
        pth_path, scalers_path = None, None
        for step in train_from_csv_lstm(csv.name, name):
            if isinstance(step, tuple) and len(step) == 5:
                last_text, _model_id, _col, pth_path, scalers_path = step
            else:
                last_text = step
                yield last_text, models, None, None

        clear_model_cache()
        updated = list_available_models()
        yield last_text, updated, pth_path, scalers_path
    except Exception as e:
        yield f"❌ 训练失败: {str(e)}", list_available_models(), None, None


def update_dropdown(choices):
    return gr.update(choices=choices, value=choices[0] if choices else "default")


def generate_csv_example():
    example_data = {
        "chain_id": [1, 1, 1, 2, 2, 2, 2, 2],
        "stage_idx": [1, 2, 3, 1, 2, 3, 4, 5],
        "W_um": [1.0, 1.2, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0],
        "CL_fF": [5.0, 6.0, 7.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        "VDD_V": [1.0, 1.0, 1.0, 1.2, 1.2, 1.2, 1.2, 1.2],
        "stage_delay_ps": [4.2, 4.1, 4.2, 3.0, 3.0, 3.0, 3.0, 3.0],
        "total_delay_ps": [12.5, 12.5, 12.5, 15.0, 15.0, 15.0, 15.0, 15.0],
    }
    df = pd.DataFrame(example_data)
    temp_file = os.path.join(UPLOAD_DIR, "csv_example_long.csv")
    df.to_csv(temp_file, index=False)
    return temp_file


PREDICT_EXAMPLES = [
    ["2.0,2.0,2.0,2.0,2.0", "5.0,5.0,5.0,5.0,5.0", "1.0,1.0,1.0,1.0,1.0", 1e-12],
    ["1.0,1.2,1.5", "5.0,6.0,7.0", "1.0,1.0,1.0", 1e-12],
    ["2.0,2.1,2.2,2.3", "8.0,8.5,9.0,9.5", "1.0,1.05,1.1,1.15", 1e-12],
]


with gr.Blocks(title="晶体管延迟预测", fill_height=True) as demo:
    gr.HTML(
        """
        <div id="app-header">
            <h1>晶体管反相器链 · 延迟预测</h1>
            <p>LSTM per-stage 建模 · 支持非均匀级联输入 · 在线训练自定义模型</p>
        </div>
        """
    )

    gr.HTML(
        """
        <div class="tag-row">
            <span>W · μm</span><span>CL · fF</span><span>VDD · V</span>
            <span>延迟 · ps</span><span>不确定性区间</span>
        </div>
        """
    )

    models_state = gr.State(list_available_models())

    with gr.Row(equal_height=True):
        with gr.Column(scale=4):
            model_dropdown = gr.Dropdown(
                label="当前模型",
                choices=list_available_models(),
                value="default",
                info="default = 预训练模型；训练完成后在此切换",
                elem_classes=["model-bar"],
            )
        with gr.Column(scale=1, min_width=120):
            refresh_btn = gr.Button("刷新列表", variant="secondary", size="sm")

    with gr.Tabs():
        with gr.Tab("延迟预测", id="predict"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, elem_classes=["panel-card"]):
                    gr.Markdown("#### 输入参数")
                    gr.Markdown(
                        "各级参数用**英文逗号**分隔，长度须一致。"
                        " 例如 5 级链：`2.0,2.0,2.0,2.0,2.0`"
                    )
                    w_input = gr.Textbox(
                        label="W — 晶体管宽度 (μm)",
                        value="2.0,2.0,2.0,2.0,2.0",
                        placeholder="2.0,2.1,2.2,...",
                    )
                    cl_input = gr.Textbox(
                        label="CL — 负载电容 (fF)",
                        value="5.0,5.0,5.0,5.0,5.0",
                        placeholder="5.0,6.0,7.0,...",
                    )
                    vdd_input = gr.Textbox(
                        label="VDD — 电源电压 (V)",
                        value="1.0,1.0,1.0,1.0,1.0",
                        placeholder="1.0,1.0,1.05,...",
                    )
                    k_input = gr.Number(
                        label="换算因子 k（相对单位 → 秒）",
                        value=1e-12,
                        precision=12,
                        info="总延迟 × k = 秒",
                    )
                    with gr.Accordion("参数说明", open=False):
                        gr.Markdown(
                            """
| 符号 | 含义 | 单位 |
|------|------|------|
| W | 每级晶体管宽度 | μm |
| CL | 每级负载电容 | fF |
| VDD | 每级电源电压 | V |
| 输出 | 延迟 | ps |

模型输出包含：各级预测均值、总延迟点估计、5%–95% 置信区间，以及与简化理论式 `CL/(W×VDD)` 的对比。
                            """
                        )
                    gr.Examples(
                        examples=PREDICT_EXAMPLES,
                        inputs=[w_input, cl_input, vdd_input, k_input],
                        label="快速示例",
                    )
                    run_btn = gr.Button(
                        "运行预测",
                        variant="primary",
                        elem_classes=["run-btn"],
                    )

                with gr.Column(scale=1, elem_classes=["panel-card"]):
                    gr.Markdown("#### 预测结果")
                    output = gr.Markdown(value="*点击「运行预测」查看结果*")
                    with gr.Row():
                        stage_plot = gr.Plot(
                            label="各级延迟对比",
                            value=_empty_plot("运行预测后显示图表"),
                        )
                        total_plot = gr.Plot(
                            label="总延迟置信区间",
                            value=_empty_plot("运行预测后显示图表"),
                        )

            run_btn.click(
                run_predict,
                [model_dropdown, w_input, cl_input, vdd_input, k_input],
                [output, stage_plot, total_plot],
            )

        with gr.Tab("训练模型", id="train"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, elem_classes=["panel-card"]):
                    gr.Markdown("#### 上传训练数据")
                    gr.Markdown(
                        "**长格式 CSV（推荐）**：`chain_id`, `stage_idx`, `W_um`, `CL_fF`, "
                        "`VDD_V`, `stage_delay_ps`（可选）, `total_delay_ps`"
                    )
                    csv_upload = gr.File(
                        label="CSV 文件",
                        file_types=[".csv"],
                        file_count="single",
                    )
                    model_name_input = gr.Textbox(
                        label="自定义模型名（可选）",
                        placeholder="留空则自动生成 UUID",
                    )
                    with gr.Row():
                        example_btn = gr.Button("下载 CSV 范例", variant="secondary")
                        train_btn = gr.Button("开始训练", variant="primary", elem_classes=["run-btn"])
                    example_file = gr.File(label="范例文件", interactive=False, visible=False)

                with gr.Column(scale=1, elem_classes=["panel-card"]):
                    gr.Markdown("#### 训练日志")
                    train_output = gr.Textbox(
                        label="",
                        lines=16,
                        max_lines=24,
                        show_label=False,
                        placeholder="训练进度将在此实时显示…",
                        elem_id="train-output",
                    )
                    gr.Markdown("##### 训练产物下载")
                    with gr.Row():
                        model_file = gr.File(label="模型 (.pth)", interactive=False)
                        scalers_file = gr.File(label="Scalers (.pkl)", interactive=False)

            train_btn.click(
                train_gen,
                [csv_upload, model_name_input, models_state],
                [train_output, models_state, model_file, scalers_file],
            ).then(update_dropdown, models_state, model_dropdown)

            example_btn.click(generate_csv_example, [], example_file).then(
                lambda f: gr.update(visible=True, value=f),
                example_file,
                example_file,
            )

    refresh_btn.click(refresh_model_list, [], [model_dropdown, models_state])

if __name__ == "__main__":
    demo.launch(theme=APP_THEME, css=CUSTOM_CSS)
