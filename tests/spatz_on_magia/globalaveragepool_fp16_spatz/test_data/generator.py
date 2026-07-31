# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Input data and golden model for the FP16 Spatz globalaveragepool test.

The kernel sums each channel with an *ordered* reduction (vfredosum) and then
divides by H*W in FP16, so the golden model sums in the same order with FP16
rounding at every step and the results must be exact. H*W stays below VLMAX
(256 @ e16/m8, VLEN = 512) so that the kernel takes a single vector chunk.

NOTE: H*W is deliberately ODD here. Channel i is read from src + i*H*W, so with an odd
H*W every other channel starts on a 2-byte address, and the Spatz VLSU corrupts vector
accesses that are not 4-byte aligned. The kernel detects that and falls back to a scalar
ordered sum, so this shape runs both paths in one go - as ResNet18's 7x7 pool does.
Set IN_H/IN_W to an even product to exercise only the vector path.
"""

import os

import numpy as np

IN_N = 2
IN_C = 9
IN_H = 3
IN_W = 5          # odd product, see the note above

def fmt(x):
    t = f"{float(x):.9g}"
    if "." not in t and "e" not in t and "n" not in t:
        t += ".0"
    return t + "f"


def emit(f, name, arr):
    v = [fmt(x) for x in np.asarray(arr).flatten()]
    f.write(f"static const float16 {name}[] = {{\n")
    for i in range(0, len(v), 8):
        f.write("    " + ", ".join(v[i:i + 8]) + ",\n")
    f.write("};\n\n")


def open_data(defines):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.h")
    f = open(path, "w")
    f.write("// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.\n")
    f.write("// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n")
    f.write("// SPDX-License-Identifier: Apache-2.0\n\n")
    f.write("/* Automatically generated header file, do not edit by hand. */\n")
    f.write("/* Regenerate with: python3 generator.py */\n")
    f.write("#ifndef DATA_H_\n#define DATA_H_\n\n")
    for k, v in defines.items():
        f.write(f"#define {k:<12} {v}\n")
    f.write("\n/* clang-format off */\n")
    return f


def close_data(f):
    f.write("/* clang-format on */\n\n#endif /* DATA_H_ */\n")
    f.close()


def main():
    rng = np.random.default_rng(0)

    X = rng.uniform(-2.0, 2.0, (IN_N, IN_C, IN_H, IN_W)).astype(np.float16)

    flat = X.reshape(IN_N * IN_C, IN_H * IN_W)
    G = np.zeros(IN_N * IN_C, dtype=np.float16)
    for c in range(flat.shape[0]):
        acc = np.float16(0.0)
        for v in flat[c]:                # vfredosum: ascending, FP16 each step
            acc = (acc + v).astype(np.float16)
        G[c] = (acc / np.float16(IN_H * IN_W)).astype(np.float16)

    f = open_data({"IN_N": IN_N, "IN_C": IN_C, "IN_H": IN_H, "IN_W": IN_W,
                   "OUT_LEN": IN_N * IN_C, "ULP_TOLL": 0})
    emit(f, "X", X)
    emit(f, "G", G)
    close_data(f)
    print(f"data.h generated [{IN_N}x{IN_C}x{IN_H}x{IN_W}]")


if __name__ == "__main__":
    main()
