# app.py — Gradio UI
import os

import gradio as gr
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


def list_available_models():
    names = ["default"]
    if os.path.isdir(MODEL_DIR):
        for fname in os.listdir(MODEL_DIR):
            if fname.endswith(".pth"):
                names.append(os.path.splitext(fname)[0])
    return sorted(list(dict.fromkeys(names)))


def run_predict(selected_model, w_str, cl_str, vdd_str, k):
    try:
        w_list, cl_list, vdd_list = parse_stage_lists(w_str, cl_str, vdd_str)
        if selected_model == "default":
            pth, pkl = DEFAULT_MODEL_PTH, DEFAULT_SCALERS_PKL
        else:
            pth = os.path.join(MODEL_DIR, f"{selected_model}.pth")
            pkl = os.path.join(MODEL_DIR, f"{selected_model}_scalers.pkl")
        result = predict_chain(w_list, cl_list, vdd_list, pth, pkl)
        return format_prediction_report(result, k=float(k))
    except Exception as e:
        err = str(e)
        if "shape" in err.lower() or "size mismatch" in err.lower() or "Missing key" in err:
            return (
                f"❌ 无法加载模型（参数形状不匹配或文件损坏）:\n{err}\n\n"
                "请用当前训练代码重新训练模型或选择其他模型。"
            )
        return f"❌ 出错: {err}"


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


with gr.Blocks() as demo:
    gr.Markdown("# 晶体管延迟预测与用户训练模型")
    gr.Markdown(
        "默认单位：\n"
        "- W（晶体管宽度 per stage）：μm (逗号分隔, e.g., 2.0,2.1,2.2)\n"
        "- CL（负载电容 per stage）：fF\n"
        "- VDD（电源电压 per stage）：V\n"
        "- 延迟：ps\n"
        "支持非均匀 per-stage 输入！"
    )

    models_state = gr.State(list_available_models())
    model_dropdown = gr.Dropdown(label="选择模型", choices=list_available_models(), value="default")

    with gr.Tab("晶体管延迟预测"):
        w_input = gr.Textbox(label="W per stage (comma sep)", value="2.0,2.0,2.0,2.0,2.0")
        cl_input = gr.Textbox(label="CL per stage (comma sep)", value="5.0,5.0,5.0,5.0,5.0")
        vdd_input = gr.Textbox(label="VDD per stage (comma sep)", value="1.0,1.0,1.0,1.0,1.0")
        k_input = gr.Number(label="换算因子 k", value=1e-12, precision=12)
        run_btn = gr.Button("运行预测")
        output = gr.Textbox(label="输出结果", lines=18)
        run_btn.click(run_predict, [model_dropdown, w_input, cl_input, vdd_input, k_input], output)

    with gr.Tab("上传 CSV 并训练新模型"):
        csv_upload = gr.File(label="上传 CSV (长格式优先)", file_types=[".csv"])
        model_name_input = gr.Textbox(label="自定义模型名（可选）")
        train_btn = gr.Button("训练新模型")
        train_output = gr.Textbox(label="训练结果", lines=12)
        model_file = gr.File(label="下载模型 PTH 文件", interactive=False)
        scalers_file = gr.File(label="下载 Scalers PKL 文件", interactive=False)
        example_btn = gr.Button("下载 CSV 范例 (长格式)")
        example_file = gr.File(label="CSV 范例", interactive=False)

        train_btn.click(
            train_gen,
            [csv_upload, model_name_input, models_state],
            [train_output, models_state, model_file, scalers_file],
        ).then(update_dropdown, models_state, model_dropdown)
        example_btn.click(generate_csv_example, [], example_file)

if __name__ == "__main__":
    demo.launch()
