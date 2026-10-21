# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

import numpy as np


def positive_int(value):
    ival = int(value)
    if ival <= 0:
        raise argparse.ArgumentTypeError(f"Value must be positive ({value}).")
    return ival


def parse_args():
    parser = argparse.ArgumentParser(description="Generator of Input Data and Golden Model for the SiLU test (ONNX Swish, alpha = 1)")

    parser.add_argument("N", type=positive_int, help="Number of elements")

    parser.add_argument("--range", type=float, default=20.0, help="X in U(-range, range) (default 20)")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# Golden in float64, correctly rounded to fp16: onnxruntime's Swish (fp16 and fp32) is off by up to
# ~24 ULP for x < -10, the kernel is not.
def golden_silu(X):
    x = X.astype(np.float64)
    return (x / (1.0 + np.exp(-x))).astype(np.float16)


def format_array(array):
    return "{ " + ", ".join(f"{float(x):.9e}f" for x in array.flatten()) + " }"


def generate_header_file(args, X, G, filename="data.h"):
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

        f.write(f"#define TENSOR_LEN {args.N}\n\n")

        f.write(f"static const float16 X[] = {format_array(X)};\n")
        f.write(f"static const float16 G[] = {format_array(G)};\n\n")

        f.write("#endif  /* DATA_H_ */\n")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    X = rng.uniform(-args.range, args.range, args.N).astype(np.float16)
    G = golden_silu(X)

    generate_header_file(args, X, G)

    print(f"File 'data.h' successfully generated with [N:{args.N}], X in U(-{args.range}, {args.range})")


if __name__ == "__main__":
    main()
