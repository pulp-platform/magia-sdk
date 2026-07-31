# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Input data and golden model for the FP16 Spatz relu kernel test.

max(x, 0) is exact in FP16, so the test requires bit-exact results.
"""

import argparse
import os

import numpy as np

# Neither a multiple of the tile count (1, 4, 16, 64) nor of Spatz's VLMAX for
# e16/m8 (256 elements @ VLEN=512): exercises the uneven shard split and the
# vector loop tail.
DEFAULT_LEN = 902

VALUE_RANGE = 4.0


def parse_args():
    parser = argparse.ArgumentParser(description="Generator for the FP16 Spatz relu kernel test")
    parser.add_argument("length", type=int, nargs="?", default=DEFAULT_LEN)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def format_value(x):
    text = f"{float(x):.9g}"
    if "." not in text and "e" not in text and "n" not in text:
        text += ".0"
    return text + "f"


def emit_array(f, name, array):
    values = [format_value(x) for x in np.asarray(array).flatten()]
    f.write(f"static const float16 {name}[] = {{\n")
    for i in range(0, len(values), 8):
        f.write("    " + ", ".join(values[i:i + 8]) + ",\n")
    f.write("};\n\n")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    X = rng.uniform(-VALUE_RANGE, VALUE_RANGE, args.length).astype(np.float16)
    G = np.maximum(X, np.float16(0)).astype(np.float16)

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.h")
    with open(path, "w") as f:
        f.write("// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.\n")
        f.write("// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n")
        f.write("// SPDX-License-Identifier: Apache-2.0\n\n")
        f.write("/* Automatically generated header file, do not edit by hand. */\n")
        f.write(f"/* Regenerate with: python3 generator.py {args.length} --seed {args.seed} */\n")
        f.write("#ifndef DATA_H_\n#define DATA_H_\n\n")
        f.write(f"#define VEC_LEN  {args.length}\n")
        f.write(f"#define OUT_LEN  {args.length}\n")
        f.write("#define ULP_TOLL 0\n\n")
        f.write("/* clang-format off */\n")
        emit_array(f, "X", X)
        emit_array(f, "G", G)
        f.write("/* clang-format on */\n\n")
        f.write("#endif /* DATA_H_ */\n")

    print(f"data.h generated [LEN:{args.length}, SEED:{args.seed}]")


if __name__ == "__main__":
    main()
