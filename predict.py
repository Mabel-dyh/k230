# predict.py — CLI for chain delay prediction (uses shared inference module)
import argparse
import os

from inference import (
    DEFAULT_MODEL_PTH,
    DEFAULT_SCALERS_PKL,
    format_prediction_report,
    predict_chain,
)


def main():
    parser = argparse.ArgumentParser(description="反相器链延迟预测 CLI")
    parser.add_argument("--W", required=True, help="每级 W (μm)，逗号分隔，如 2.0,2.0,2.0")
    parser.add_argument("--CL", required=True, help="每级 CL (fF)，逗号分隔")
    parser.add_argument("--VDD", required=True, help="每级 VDD (V)，逗号分隔")
    parser.add_argument("--model", default=DEFAULT_MODEL_PTH, help="模型 .pth 路径")
    parser.add_argument("--scalers", default=DEFAULT_SCALERS_PKL, help="scalers .pkl 路径")
    parser.add_argument("--k", type=float, default=1e-12, help="换算为秒的因子")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"模型文件不存在: {args.model}")
    if not os.path.exists(args.scalers):
        raise FileNotFoundError(f"Scalers 文件不存在: {args.scalers}")

    w_list = [float(x.strip()) for x in args.W.split(",") if x.strip()]
    cl_list = [float(x.strip()) for x in args.CL.split(",") if x.strip()]
    vdd_list = [float(x.strip()) for x in args.VDD.split(",") if x.strip()]

    result = predict_chain(w_list, cl_list, vdd_list, args.model, args.scalers)
    print(format_prediction_report(result, k=args.k))


if __name__ == "__main__":
    main()
