# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Input data and golden model for the FP16 Spatz conv2d kernel test.

The golden model reproduces the kernel's exact summation structure, which matters
much more than it sounds: each output pixel sums c_in * kh * kw products of
similar magnitude and opposite signs, so in FP16 a different summation order does
not just shift the last bit - it can flip the sign of a near-zero result.

The kernel keeps one FP16 accumulator per kernel *column* (the vector lanes),
accumulates products into them with fused multiply-adds over (in-channel, kernel
row), and finally folds the lanes into the bias with an ordered reduction
(vfredosum.vs v0, v8, v0, seeded with the broadcast bias). Reproducing that
grouping makes the test bit-exact. Windows clipped by padding shorten both the
source and the weight run (the kernel offsets the weights by ker_start).

Output channels are the sharded dimension: the kernel splits n * c_out over tiles.
"""

# NOTE on the configuration below: a 2x2 kernel with stride 2 and no padding is
# chosen so that every vector access the kernel makes is 4-byte aligned. The Spatz
# VLSU corrupts misaligned vector accesses (the kernel authors documented this for
# stores - see the disabled alignment guard in
# kernels/spatz_fp16/transpose/spatz_task/transpose_fp16_spatz_task.c - and it
# affects vle16.v loads just as much). conv2d loads both the source window and the
# weight row with vle16.v at offset w_win_start and kh * kernel_w, so an odd
# kernel_w or an odd stride puts those loads on 2-byte addresses: with the usual
# 3x3 / stride 1 / pad 1 configuration this test fails on ~100% of the output
# (1150 of 1152 elements), while this aligned configuration is bit-exact.

import os

import numpy as np

IN_N     = 1
IN_C     = 3
IN_H     = 8
IN_W     = 8
OUT_C    = 18          # n * OUT_C is what gets sharded over the tiles
KERNEL_H = 2
KERNEL_W = 2
STRIDE_H = 2
STRIDE_W = 2
PAD_H    = 0
PAD_W    = 0
GROUPS   = 1
HAS_BIAS = 1

OUT_H = (IN_H + 2 * PAD_H - KERNEL_H) // STRIDE_H + 1
OUT_W = (IN_W + 2 * PAD_W - KERNEL_W) // STRIDE_W + 1

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


def window(out_idx, stride, pad, shape, in_len):
    """compute_window_boundaries(): clip the window, offset the weights"""
    logical_start = out_idx * stride - pad
    start = max(logical_start, 0)
    ker_start = -logical_start if logical_start < 0 else 0
    last = min(logical_start + shape, in_len)
    return start, max(last - start, 0), ker_start


def main():
    rng = np.random.default_rng(0)

    X = rng.uniform(-1.0, 1.0, (IN_N, IN_C, IN_H, IN_W)).astype(np.float16)
    W = rng.uniform(-0.5, 0.5, (OUT_C, IN_C // GROUPS, KERNEL_H, KERNEL_W)).astype(np.float16)
    Bs = rng.uniform(-0.5, 0.5, OUT_C).astype(np.float16)

    G = np.zeros((IN_N, OUT_C, OUT_H, OUT_W), dtype=np.float16)
    c_in_g = IN_C // GROUPS
    c_out_g = OUT_C // GROUPS
    for n in range(IN_N):
        for oc in range(OUT_C):
            g = oc // c_out_g
            for oh in range(OUT_H):
                h_start, h_len, h_ker = window(oh, STRIDE_H, PAD_H, KERNEL_H, IN_H)
                for ow in range(OUT_W):
                    w_start, w_len, w_ker = window(ow, STRIDE_W, PAD_W, KERNEL_W, IN_W)

                    # one accumulator per kernel column = one vector lane
                    lane = [np.float16(0.0)] * w_len
                    for ic in range(c_in_g):
                        for kh in range(h_len):
                            for j in range(w_len):      # vfmacc.vv, lane-wise
                                lane[j] = fma16(lane[j],
                                                X[n, g * c_in_g + ic, h_start + kh, w_start + j],
                                                W[oc, ic, h_ker + kh, w_ker + j])

                    # vfredosum.vs seeded with the bias, ascending lanes
                    acc = np.float16(Bs[oc]) if HAS_BIAS else np.float16(0.0)
                    for j in range(w_len):
                        acc = (acc + lane[j]).astype(np.float16)
                    G[n, oc, oh, ow] = acc

    f = open_data({"IN_N": IN_N, "IN_C": IN_C, "IN_H": IN_H, "IN_W": IN_W,
                   "OUT_C": OUT_C, "OUT_H": OUT_H, "OUT_W": OUT_W,
                   "KERNEL_H": KERNEL_H, "KERNEL_W": KERNEL_W,
                   "STRIDE_H": STRIDE_H, "STRIDE_W": STRIDE_W,
                   "PAD_H": PAD_H, "PAD_W": PAD_W,
                   "GROUPS": GROUPS, "HAS_BIAS": HAS_BIAS,
                   "OUT_LEN": G.size, "ULP_TOLL": 0})
    emit(f, "X", X)
    emit(f, "WEIGHTS", W)
    emit(f, "BIASES", Bs)
    emit(f, "G", G)
    close_data(f)
    print(f"data.h generated [{IN_N}x{IN_C}x{IN_H}x{IN_W} -> {OUT_C}x{OUT_H}x{OUT_W}]")


if __name__ == "__main__":
    main()
