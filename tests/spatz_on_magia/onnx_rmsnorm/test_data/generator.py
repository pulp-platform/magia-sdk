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
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for ONNX RMSNormalization test")

    parser.add_argument("R", type=positive_int, help="Number of rows (tokens)")
    parser.add_argument("W", type=positive_int, help="Row length (normalized axis, hidden size)")

    parser.add_argument("--epsilon", type=float, default=1e-5, help="Epsilon (default 1e-5, SmolLM2)")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def run_onnx_rmsnorm(X, scale, epsilon):
    X_info = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT16, X.shape)
    S_info = onnx.helper.make_tensor_value_info('S', onnx.TensorProto.FLOAT16, scale.shape)
    Y_info = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT16, X.shape)

    opset = onnx.helper.make_operatorsetid("", 23)
    node_def = onnx.helper.make_node('RMSNormalization', ['X', 'S'], ['Y'], axis=-1, epsilon=epsilon)
    graph_def = onnx.helper.make_graph([node_def], 'onnx-rmsnorm-test', [X_info, S_info], [Y_info])
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-generator', opset_imports=[opset])
    model_def.ir_version = 10

    sess = ort.InferenceSession(model_def.SerializeToString())
    return sess.run(None, {'X': X, 'S': scale})[0]


def format_array(array):
    return "{ " + ", ".join(f"{float(x):.9e}f" for x in array.flatten()) + " }"


def generate_header_file(args, X, scale, G, filename="data.h"):
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

        f.write(f"#define ROWS {args.R}\n")
        f.write(f"#define WIDTH {args.W}\n\n")

        f.write(f"static const float16 epsilon = {float(np.float16(args.epsilon)):.9e}f;\n")
        f.write(f"static const float16 X[] = {format_array(X)};\n")
        f.write(f"static const float16 scale[] = {format_array(scale)};\n")
        f.write(f"static const float16 G[] = {format_array(G)};\n\n")

        f.write("#endif  /* DATA_H_ */\n")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    X = rng.standard_normal((args.R, args.W)).astype(np.float16)
    scale = rng.uniform(0.5, 1.5, args.W).astype(np.float16)
    # the kernel reads epsilon as fp16: the golden uses the same value
    G = run_onnx_rmsnorm(X, scale, float(np.float16(args.epsilon)))

    generate_header_file(args, X, scale, G)

    print(f"File 'data.h' successfully generated with [R:{args.R}, W:{args.W}], epsilon {args.epsilon}")


if __name__ == "__main__":
    main()
