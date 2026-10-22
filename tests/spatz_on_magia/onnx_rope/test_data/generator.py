# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

import numpy as np
import onnx
import onnxruntime as ort


def positive_int(value):
    ival = int(value)
    if ival <= 0:
        raise argparse.ArgumentTypeError(f"Value must be positive ({value}).")
    return ival


def parse_args():
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for ONNX RotaryEmbedding test (rotate-half, interleaved = 0)")

    parser.add_argument("S", type=positive_int, help="Number of rows (tokens)")
    parser.add_argument("H", type=positive_int, help="Number of heads")
    parser.add_argument("D", type=positive_int, help="Head size (even)")

    parser.add_argument("--pos0", type=int, default=113, help="Position of the first token (default 113)")
    parser.add_argument("--theta", type=float, default=100000.0, help="RoPE base (default 100000, SmolLM2)")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def run_onnx_rope(X, cos_cache, sin_cache, pos, n_heads):
    X_info = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT16, X.shape)
    C_info = onnx.helper.make_tensor_value_info('C', onnx.TensorProto.FLOAT16, cos_cache.shape)
    S_info = onnx.helper.make_tensor_value_info('S', onnx.TensorProto.FLOAT16, sin_cache.shape)
    P_info = onnx.helper.make_tensor_value_info('P', onnx.TensorProto.INT64, pos.shape)
    Y_info = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT16, X.shape)

    opset = onnx.helper.make_operatorsetid("", 23)
    node_def = onnx.helper.make_node('RotaryEmbedding', ['X', 'C', 'S', 'P'], ['Y'], num_heads=n_heads, interleaved=0)
    graph_def = onnx.helper.make_graph([node_def], 'onnx-rope-test', [X_info, C_info, S_info, P_info], [Y_info])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])
    model_def.ir_version = 10

    sess = ort.InferenceSession(model_def.SerializeToString())
    return sess.run(None, {'X': X, 'C': cos_cache, 'S': sin_cache, 'P': pos})[0]


def format_array(array):
    return "{ " + ", ".join(f"{float(x):.9e}f" for x in array.flatten()) + " }"


def generate_header_file(args, X, cos_tab, sin_tab, G, filename="data.h"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)

    with open(filepath, "w") as f:
        f.write("// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.\n")
        f.write("// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n")
        f.write("// SPDX-License-Identifier: Apache-2.0\n")
        f.write("\n")
        f.write("/* Automatically generated header file for Spatz ONNX testing */\n")
        f.write("#ifndef DATA_H_\n")
        f.write("#define DATA_H_\n\n")
        f.write(f"#define ROWS {args.S}\n")
        f.write(f"#define HEADS {args.H}\n")
        f.write(f"#define HEAD_DIM {args.D}\n\n")

        f.write(f"static const float16 X[] = {format_array(X)};\n")
        f.write(f"static const float16 cos_tab[] = {format_array(cos_tab)};\n")
        f.write(f"static const float16 sin_tab[] = {format_array(sin_tab)};\n")
        f.write(f"static const float16 G[] = {format_array(G)};\n\n")

        f.write("#endif  /* DATA_H_ */\n")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    assert args.D % 2 == 0

    X = rng.standard_normal((1, args.S, args.H * args.D)).astype(np.float16)
    # cos/sin caches over all positions up to the last token, as in a Llama model
    max_pos = args.pos0 + args.S
    inv_freq = 1.0 / (args.theta ** (np.arange(0, args.D, 2, dtype=np.float64) / args.D))
    angles = np.arange(max_pos, dtype=np.float64)[:, None] * inv_freq[None, :]
    cos_cache = np.cos(angles).astype(np.float16)
    sin_cache = np.sin(angles).astype(np.float16)
    pos = (args.pos0 + np.arange(args.S, dtype=np.int64))[None, :]

    G = run_onnx_rope(X, cos_cache, sin_cache, pos, args.H)

    # the kernel takes the cos/sin rows already gathered at each token's position
    generate_header_file(args, X, cos_cache[pos[0]], sin_cache[pos[0]], G)

    print(f"File 'data.h' successfully generated with [S:{args.S}, H:{args.H}, D:{args.D}], positions {args.pos0}..{max_pos - 1}")


if __name__ == "__main__":
    main()
