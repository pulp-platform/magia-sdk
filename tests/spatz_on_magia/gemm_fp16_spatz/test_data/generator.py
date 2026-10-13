# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Input data and golden model for the FP16 Spatz gemm kernel test.

Y = alpha * A @ B + beta * C. The kernel accumulates with vfmacc, a fused
multiply-add over ascending k, then scales by alpha and folds in beta * C with
another fused multiply-add. The golden model performs exactly those operations
with one rounding each, so the results must be exact.
"""

# NOTE: N is deliberately ODD here. The Spatz VLSU corrupts vector accesses to
# addresses that are not 4-byte aligned, and the B, C and Y rows all step by N, so an
# odd N puts every other row on a 2-byte address. This shape therefore exercises the
# kernel's alignment guard and its scalar fallback, which produces the same roundings
# (ascending-k fused multiply-adds). Without that guard this configuration is wrong on
# a large fraction of the elements.

# Set DIM_N to 10 to make every access aligned and exercise the vector path
# instead of the fallback.

import os

import numpy as np

DIM_M  = 18
DIM_K  = 7
DIM_N  = 9
ALPHA  = 1.5
BETA   = 0.5
TRANS_A = 0
TRANS_B = 0

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


def fma16(acc, a, b):
    """One FP16 fused multiply-add: exact product, single rounding (vfmacc)"""
    return np.float16(float(acc) + float(a) * float(b))


def main():
    rng = np.random.default_rng(0)

    A = rng.uniform(-1.5, 1.5, (DIM_M, DIM_K)).astype(np.float16)
    B = rng.uniform(-1.5, 1.5, (DIM_K, DIM_N)).astype(np.float16)
    C = rng.uniform(-1.0, 1.0, (DIM_M, DIM_N)).astype(np.float16)
    alpha = np.float16(ALPHA)
    beta = np.float16(BETA)

    G = np.zeros((DIM_M, DIM_N), dtype=np.float16)
    for m in range(DIM_M):
        for n in range(DIM_N):
            acc = np.float16(0.0)
            for k in range(DIM_K):                 # vfmacc.vf, ascending k
                acc = fma16(acc, A[m, k], B[k, n])
            acc = (acc * alpha).astype(np.float16)  # vfmul.vf
            acc = fma16(acc, beta, C[m, n])         # vfmacc.vf
            G[m, n] = acc

    f = open_data({"DIM_M": DIM_M, "DIM_K": DIM_K, "DIM_N": DIM_N,
                   "ALPHA": f"{ALPHA}f", "BETA": f"{BETA}f",
                   "TRANS_A": TRANS_A, "TRANS_B": TRANS_B,
                   "OUT_LEN": DIM_M * DIM_N, "ULP_TOLL": 0})
    emit(f, "A", A)
    emit(f, "B", B)
    emit(f, "C", C)
    emit(f, "G", G)
    close_data(f)
    print(f"data.h generated [{DIM_M}x{DIM_K} @ {DIM_K}x{DIM_N}]")


if __name__ == "__main__":
    main()
