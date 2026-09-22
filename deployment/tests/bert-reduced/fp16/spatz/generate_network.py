import argparse
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper
from onnxsim import simplify
from transformers import BertModel


MODEL_ID = "google/bert_uncased_L-2_H-128_A-2"
DOWNLOAD_DIR = "download"
SEQ = 16


def parse_args():
    parser = argparse.ArgumentParser(description="Generator for the reduced (Google BERT-Tiny) encoder test")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the random input (default: 0)")
    return parser.parse_args()


def load(download_dir):
    os.makedirs(download_dir, exist_ok=True)
    model = BertModel.from_pretrained(MODEL_ID, add_pooling_layer=True, attn_implementation="eager", cache_dir=download_dir)
    return model.eval()


class _Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs_embeds):
        return self.model(inputs_embeds=inputs_embeds).pooler_output


def export_onnx(model, onnx_path):
    encoder = _Encoder(model)
    dummy = torch.randn(1, SEQ, model.config.hidden_size)
    torch.onnx.export(encoder, (dummy,), onnx_path, input_names=["inputs_embeds"], output_names=["out"], opset_version=20, do_constant_folding=True, dynamo=False)
    return onnx_path


def simplify_onnx(onnx_path, hidden):
    model, ok = simplify(onnx.load(onnx_path), overwrite_input_shapes={"inputs_embeds": [1, SEQ, hidden]})
    assert ok, "onnx-simplifier failed to simplify the BERT encoder"
    return model


def convert(model, work_dir):
    onnx_path = os.path.join(work_dir, "bert_fp32.onnx")
    export_onnx(model, onnx_path)
    return simplify_onnx(onnx_path, model.config.hidden_size)


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
    hidden = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
    X = np.random.randn(1, SEQ, hidden).astype(np.float16)
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


    print(f"Deployment files generated (Google BERT-Tiny encoder, fp16) "
          f"[input: {onnx_model.graph.input[0].name} (1,{SEQ},{X.shape[2]}) -> {tuple(Y.shape)}]")


if __name__ == "__main__":
    main()
