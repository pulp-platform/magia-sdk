import os
import onnx
import argparse

import numpy as np
import onnxruntime as ort

from onnx import numpy_helper
from ultralytics import YOLO

DOWNLOAD_DIR = "download"
MODEL = "yolov8n"
IMG = 64


def parse_args():
    parser = argparse.ArgumentParser(description="Generator for the reduced (Ultralytics YOLOv8n) detector test")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the random input (default: 0)")
    return parser.parse_args()


def export_onnx(download_dir):
    os.makedirs(download_dir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(download_dir)
    try:
        model = YOLO(MODEL + ".pt")
        onnx_path = os.path.abspath(model.export(format="onnx", imgsz=IMG, opset=19, simplify=True, dynamic=False))
    finally:
        os.chdir(cwd)
    return onnx.load(onnx_path)


def cast_to_fp16(model):
    from onnxconverter_common import float16
    # op_block_list=[] converts every op to fp16 (default keeps Resize/etc. in fp32, which inserts
    # mismatched casts at the Resize boundary); MAGIA's Resize kernel is fp16, so we want it in fp16.
    return float16.convert_float_to_float16(model, keep_io_types=False, op_block_list=[])


def flush_subnormals(model):
    """Zero every fp16 subnormal weight (|w| < 2^-14). Deeploy's float type-checker
    rejects subnormal constants. The change is numerically negligible."""
    smallest_normal = np.float16(2.0**-14)
    for idx, init in enumerate(model.graph.initializer):
        if init.data_type != onnx.TensorProto.FLOAT16:
            continue
        arr = numpy_helper.to_array(init).copy()
        arr[np.abs(arr) < smallest_normal] = np.float16(0)
        model.graph.initializer[idx].CopyFrom(numpy_helper.from_array(arr, init.name))
    return model


def fix_resize_inputs(model):
    """Resize's roi/scales inputs are float32 by ONNX spec; the fp16 conversion wrongly casts
    them to fp16, which is an invalid graph. Cast just those constants back to float32 (the data
    input and output stay fp16, which is what MAGIA's fp16 Resize kernel expects)."""
    targets = set()
    for n in model.graph.node:
        if n.op_type == "Resize":
            for idx in (1, 2):
                if idx < len(n.input) and n.input[idx]:
                    targets.add(n.input[idx])

    for n in model.graph.node:
        if n.op_type == "Constant" and n.output[0] in targets:
            for a in n.attribute:
                if a.name == "value" and a.t.data_type == onnx.TensorProto.FLOAT16:
                    a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float32), a.t.name))

    for idx, init in enumerate(model.graph.initializer):
        if init.name in targets and init.data_type == onnx.TensorProto.FLOAT16:
            model.graph.initializer[idx].CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float32), init.name))

    return model


def to_fp16(model):
    model = cast_to_fp16(model)
    model = fix_resize_inputs(model)
    model = flush_subnormals(model)
    return model


def inference(model, seed):
    np.random.seed(seed)
    X = np.random.randn(1, 3, IMG, IMG).astype(np.float16)
    in_name = model.graph.input[0].name
    Y = ort.InferenceSession(model.SerializeToString()).run(None, {in_name: X})[0].astype(np.float16)
    return X, Y


def save(model, X, Y, out_dir):
    onnx.save(model, os.path.join(out_dir, "network.onnx"))
    np.savez(os.path.join(out_dir, "inputs.npz"), X=X)
    np.savez(os.path.join(out_dir, "outputs.npz"), Y=Y)


def main():
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    download_dir = os.path.join(here, DOWNLOAD_DIR)

    onnx_model = export_onnx(download_dir)
    onnx_model = to_fp16(onnx_model)
    X, Y = inference(onnx_model, args.seed)
    save(onnx_model, X, Y, here)

    print(f"Deployment files generated (Ultralytics YOLOv8n, fp16) "
          f"[input: {onnx_model.graph.input[0].name} (1,3,{IMG},{IMG}) -> {tuple(Y.shape)}]")


if __name__ == "__main__":
    main()
