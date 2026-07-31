# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Input data and golden model for the FP16 Spatz reducesum kernel test.

Reduces the middle axis of an outer x reduce x inner view. The kernel keeps one
FP16 accumulator per inner element and adds the rows in ascending order, so the
golden model accumulates the same way and the results must be exact. INNER stays
below VLMAX (256 @ e16/m8, VLEN = 512) so that the kernel takes a single chunk.
"""

# NOTE: the innermost (contiguous) dimension is deliberately ODD here. The Spatz
# VLSU corrupts vector accesses to addresses that are not 4-byte aligned, and an odd
# inner length makes every other row - and half of the accumulating loads - land on
# a 2-byte address, so this shape exercises the kernel's alignment guard and its
# scalar fallback. Without that guard this configuration is wrong on most elements.

# Set INNER to 16 to make every access aligned and exercise the vector path
# instead of the fallback.

import os

import numpy as np

OUTER  = 18
REDUCE = 5
INNER  = 17

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

    X = rng.uniform(-2.0, 2.0, (OUTER, REDUCE, INNER)).astype(np.float16)

    G = np.zeros((OUTER, INNER), dtype=np.float16)
    for o in range(OUTER):
        acc = np.zeros(INNER, dtype=np.float16)
        for r in range(REDUCE):          # same order as the kernel's row loop
            acc = (acc + X[o, r]).astype(np.float16)
        G[o] = acc

    f = open_data({"OUTER_DIM": OUTER, "REDUCE_DIM": REDUCE, "INNER_DIM": INNER,
                   "OUT_LEN": OUTER * INNER, "ULP_TOLL": 0})
    emit(f, "X", X)
    emit(f, "G", G)
    close_data(f)
    print(f"data.h generated [{OUTER}x{REDUCE}x{INNER}]")


if __name__ == "__main__":
    main()
