import os
import onnx
import tf2onnx
import argparse
import urllib.request

import numpy as np
import tensorflow as tf
import onnxruntime as ort

from collections import defaultdict
from onnx import helper, numpy_helper

MODEL_URL = ("https://raw.githubusercontent.com/mlcommons/tiny/master/"
             "benchmark/training/image_classification/trained_models/pretrainedResnet.h5")
DOWNLOAD_DIR = "download"


def parse_args():
    parser = argparse.ArgumentParser(description="Generator for the reduced (MLPerf Tiny) ResNet test")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the random input (default: 0)")
    return parser.parse_args()


def download(download_dir):
    os.makedirs(download_dir, exist_ok=True)
    h5_path = os.path.join(download_dir, "pretrainedResnet.h5")
    if not os.path.exists(h5_path):
        urllib.request.urlretrieve(MODEL_URL, h5_path)
    return h5_path


def keras_to_onnx(h5_path, onnx_path):
    model = tf.keras.models.load_model(h5_path, compile=False)
    spec = (tf.TensorSpec((1, 32, 32, 3), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(model, input_signature=spec, opset=17, inputs_as_nchw=["input"], output_path=onnx_path)
    return onnx_path


def rebuild_batchnorm(model):
    """Restore the model's real BatchNormalization operator. tf2onnx lowers each
    inference-mode BatchNorm into Conv -> Mul(scale) -> Add(shift) (a per-channel
    affine); we rewrite every such Mul+Add pair back into a single BatchNormalization
    node, staying faithful to the original model's operators and letting MAGIA's
    dedicated batchnorm kernel run it as one fused op. Using mean=0, var=1, eps=0 makes
    the node compute y = scale*x + shift, numerically identical to the lowered form."""
    g = model.graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    consumers = defaultdict(list)
    for n in g.node:
        for i in n.input:
            consumers[i].append(n)

    def single_const_operand(node):
        consts = [i for i in node.input if i in init]
        tensors = [i for i in node.input if i not in init]
        return consts[0] if (len(consts) == 1 and len(tensors) == 1) else None

    replace = {}          # add-node name -> new BatchNormalization node
    drop = set()          # mul-node names to drop
    extra_init = []
    for conv in g.node:
        if conv.op_type != "Conv":
            continue
        mul_cons = consumers[conv.output[0]]
        if len(mul_cons) != 1 or mul_cons[0].op_type != "Mul":
            continue
        mul = mul_cons[0]
        s_name = single_const_operand(mul)
        if s_name is None:
            continue
        add_cons = consumers[mul.output[0]]
        if len(add_cons) != 1 or add_cons[0].op_type != "Add":
            continue
        add = add_cons[0]
        t_name = single_const_operand(add)
        if t_name is None:
            continue

        scale = init[s_name].reshape(-1).astype(np.float32)
        shift = init[t_name].reshape(-1).astype(np.float32)
        channels = scale.shape[0]
        prefix = conv.name.replace("/", "_") + "_bn"
        names = [f"{prefix}_scale", f"{prefix}_B", f"{prefix}_mean", f"{prefix}_var"]
        extra_init += [
            numpy_helper.from_array(scale, names[0]),
            numpy_helper.from_array(shift, names[1]),
            numpy_helper.from_array(np.zeros(channels, np.float32), names[2]),
            numpy_helper.from_array(np.ones(channels, np.float32), names[3]),
        ]
        replace[add.name] = helper.make_node(
            "BatchNormalization", [conv.output[0]] + names, [add.output[0]], name=prefix, epsilon=0.0)
        drop.add(mul.name)

    new_nodes = []
    for n in g.node:
        if n.name in drop:
            continue
        new_nodes.append(replace.get(n.name, n))

    used = {i for n in new_nodes for i in n.input}
    new_init = [i for i in list(g.initializer) + extra_init if i.name in used]
    new_graph = helper.make_graph(new_nodes, g.name, g.input, g.output, new_init)
    new_model = helper.make_model(new_graph, opset_imports=model.opset_import, producer_name="resnet8-reduced")
    new_model = onnx.shape_inference.infer_shapes(new_model)
    onnx.checker.check_model(new_model)
    return new_model


def convert(h5_path, work_dir):
    onnx_path = os.path.join(work_dir, "resnet8_fp32.onnx")
    keras_to_onnx(h5_path, onnx_path)
    return rebuild_batchnorm(onnx.load(onnx_path))


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
    X = np.random.randn(1, 3, 32, 32).astype(np.float16)
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

    h5_path = download(download_dir)
    model = convert(h5_path, download_dir)
    model = to_fp16(model)
    X, Y = inference(model, args.seed)
    save(model, X, Y, here)

    print(f"Deployment files generated (MLPerf Tiny ResNet-8, fp16) [input: {model.graph.input[0].name} (1,3,32,32) -> (1,10)]")


if __name__ == "__main__":
    main()
