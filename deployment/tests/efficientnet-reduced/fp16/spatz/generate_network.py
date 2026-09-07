import os
import onnx
import argparse

import numpy as np
import torch
import timm
import onnxruntime as ort

from onnx import numpy_helper
from onnxsim import simplify

DOWNLOAD_DIR = "download"
MODEL = "efficientnet_lite0"
IMG = 8


def parse_args():
    parser = argparse.ArgumentParser(description="Generator for the reduced (timm EfficientNet-Lite0) test")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the random input (default: 0)")
    return parser.parse_args()


def load(download_dir):
    os.makedirs(download_dir, exist_ok=True)
    os.environ["HF_HOME"] = download_dir
    model = timm.create_model(MODEL, pretrained=True)
    return model.eval()


def export_onnx(model, onnx_path):
    dummy = torch.randn(1, 3, IMG, IMG)
    torch.onnx.export(model, (dummy,), onnx_path, input_names=["input"], output_names=["out"], opset_version=20, do_constant_folding=True, dynamo=False)
    return onnx_path


def simplify_onnx(onnx_path):
    model, ok = simplify(onnx.load(onnx_path), overwrite_input_shapes={"input": [1, 3, IMG, IMG]})
    assert ok, "onnx-simplifier failed to simplify the EfficientNet-Lite0 model"
    return model


def convert(model, work_dir):
    onnx_path = os.path.join(work_dir, "efficientnet_lite0_fp32.onnx")
    export_onnx(model, onnx_path)
    return simplify_onnx(onnx_path)


def cast_to_fp16(model):
    from onnxconverter_common import float16
    return float16.convert_float_to_float16(model, keep_io_types=False)


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


def to_fp16(model):
    model = cast_to_fp16(model)
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

    model = load(download_dir)
    onnx_model = convert(model, download_dir)
    onnx_model = to_fp16(onnx_model)
    X, Y = inference(onnx_model, args.seed)
    save(onnx_model, X, Y, here)

    print(f"Deployment files generated (timm EfficientNet-Lite0, fp16) "
          f"[input: {onnx_model.graph.input[0].name} (1,3,{IMG},{IMG}) -> {tuple(Y.shape)}]")


if __name__ == "__main__":
    main()
