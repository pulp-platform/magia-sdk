#!/usr/bin/env python3
# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""
Golden data for the conv2dgemm_fp16_spatz kernel test.

Regenerate with: python3 generator.py

Five convolutions are emitted over the same input, because the kernel takes a different
path for each:

  A) 3x3, stride 1, pad 1  - the iDMA im2col path. With stride_w == 1 an im2col row over
     a block of output rows is a strided rectangle of the input, so the kernel builds it
     with one 2D descriptor per (input channel, kernel tap).

  B) 3x3, stride 2, pad 1  - the scalar im2col path. With stride_w != 1 the innermost run
     is strided, which would need the iDMA's third dimension; the GVSoC magia_v3 model
     does not implement that, so the kernel falls back to the CV32.

  C) 3x3, stride 1, pad 1, group = 64  - depthwise, over the iDMA im2col.
  D) 3x3, stride 2, pad 1, group = 64  - depthwise, over the scalar im2col.

     Depthwise is the extreme of the kernel's group handling: C_in/group = C_out/group = 1,
     so K_g = 9 against K = 576, every output channel is its own group, and the group loop
     runs once per channel the tile owns. That is what mobilevit_v2 and tinyvit5m are full
     of, and it is the case the kernel used to run as a dense GEMM over a block-diagonal A.

  E) 3x3, stride 1, pad 1, group = 4, C_out = 32  - grouped with C_out/group = 8, i.e. a
     group loop whose blocks are several channels wide rather than one.

The shapes are picked to exercise the blocking rather than to be fast:

  - C_in = 64 and a 33x33 plane make one output row cost (K + 2*oc_len) * W_out * 2 =
    about 39 KB of L1, so the 896 KB budget holds 23 rows. The kernel rounds that down to
    a divisor of H_out = 33, giving 11 rows per block and 3 blocks - i.e. the weight slice
    has to stay resident across blocks while B, C and Y are refilled.
  - W_out = 33 is odd, so a block is 11 * 33 = 363 columns, which the kernel has to round
    up to 364 to keep every B/C/Y row 4-byte aligned for the Spatz VLSU. The extra column
    is computed and then dropped, which this test would catch if it were not.
  - Padding 1 on a 3x3 kernel means every edge tap is partially out of bounds, so the
    zero-filling of the invalid rows and columns is exercised on all four edges.
  - H_out = 17 in case B is also odd, so the even-rounding happens there too.

The golden replays the kernel's arithmetic exactly, so the test demands bit-exactness.
The Spatz task accumulates the GEMM over ascending k with one vfmacc per k (exact
product, single rounding), scales by alpha, then folds in beta * bias with one more
fused multiply-add - see conv2dgemm_fp16_spatz_task.c. Blocking, the output-channel
sharding and the grouping are exact partitions of that same per-element computation, so
the golden does not depend on any of them: a group's GEMM over K_g and the old dense GEMM
over a zero-padded K differ only by terms `acc = fma(0, b, acc)`, which leave acc alone.
"""

import os

import numpy as np

# --- configuration -----------------------------------------------------------

IN_N = 1
IN_C = 64
IN_H = 33
IN_W = 33

KERNEL_H = 3
KERNEL_W = 3
HAS_BIAS = 1

# Filter banks, each drawn once and shared by the cases that use it. The order matters:
# the first entry has to be drawn first so cases 1 and 2 keep the goldens they had before
# the grouped cases were added.
FILTERS = [
    # name, out_c, groups
    ("", 32, 1),      # WEIGHTS / BIASES
    ("_DW", 64, 64),  # depthwise: one filter per input channel
    ("_G4", 32, 4),   # grouped: C_in/group = 16, C_out/group = 8
]

# (tag, stride, pad, filter name)
CASES = [
    ("1", 1, 1, ""),
    ("2", 2, 1, ""),
    ("3", 1, 1, "_DW"),
    ("4", 2, 1, "_DW"),
    ("5", 1, 1, "_G4"),
]

SEED = 20260728

# --- helpers -----------------------------------------------------------------


def out_dim(in_dim, kernel, stride, pad):
    return (in_dim + 2 * pad - kernel) // stride + 1


def im2col(X_b, c_in, h_in, w_in, h_out, w_out, kh, kw, sh, sw, ph, pw, ic0=0):
    """B[k, n] with k = (ic*K_h + ki)*K_w + kj and n = oh*W_out + ow, zero outside.

    ic0 is the first input channel of the group; k is numbered from it, so a grouped
    convolution gets its own [K_g, N] matrix rather than a slice of the full one."""
    K = c_in * kh * kw
    B = np.zeros((K, h_out * w_out), dtype=np.float16)

    for ic in range(c_in):
        plane = X_b[ic0 + ic]
        for ki in range(kh):
            for kj in range(kw):
                k = (ic * kh + ki) * kw + kj
                for oh in range(h_out):
                    ih = oh * sh - ph + ki
                    if ih < 0 or ih >= h_in:
                        continue
                    for ow in range(w_out):
                        iw = ow * sw - pw + kj
                        if 0 <= iw < w_in:
                            B[k, oh * w_out + ow] = plane[ih, iw]

    return B


def gemm_fp16_exact(A, B, bias, alpha=1.0, beta=1.0):
    """
    Y = alpha * A @ B + beta * bias, accumulated the way the Spatz task does it:
    acc starts at zero, one fused multiply-add per k in ascending order with a single
    rounding to fp16 each time, then the alpha scale, then one more fused multiply-add
    for the bias. float64 intermediates keep each individual fma exact before it is
    rounded back down, which is what fmadd.h does.
    """
    M, K = A.shape
    N = B.shape[1]

    acc = np.zeros((M, N), dtype=np.float16)
    A64 = A.astype(np.float64)
    B64 = B.astype(np.float64)

    for k in range(K):
        acc = (acc.astype(np.float64) + np.outer(A64[:, k], B64[k, :])).astype(np.float16)

    if alpha != 1.0:
        acc = (acc.astype(np.float64) * np.float64(alpha)).astype(np.float16)

    if bias is not None:
        acc = (
            np.float64(beta) * bias.astype(np.float64)[:, None] + acc.astype(np.float64)
        ).astype(np.float16)

    return acc


def conv_fp16_exact(X_b, W, bias, groups, stride, pad, h_out, w_out):
    """One group at a time, exactly as the kernel runs it: the group's own [K_g, N]
    im2col against its [C_out/group, K_g] slice of the filters."""
    out_c = W.shape[0]
    cin_g = W.shape[1]
    cout_g = out_c // groups
    K_g = cin_g * KERNEL_H * KERNEL_W

    Y = np.empty((out_c, h_out * w_out), dtype=np.float16)

    for g in range(groups):
        Bmat = im2col(
            X_b, cin_g, IN_H, IN_W, h_out, w_out,
            KERNEL_H, KERNEL_W, stride, stride, pad, pad, ic0=g * cin_g,
        )
        A = W[g * cout_g : (g + 1) * cout_g].reshape(cout_g, K_g)
        b = bias[g * cout_g : (g + 1) * cout_g] if bias is not None else None

        Y[g * cout_g : (g + 1) * cout_g] = gemm_fp16_exact(A, Bmat, b)

    return Y


def fmt(x):
    t = f"{float(x):.9g}"
    if "." not in t and "e" not in t and "n" not in t:
        t += ".0"
    return t + "f"


def emit(f, name, arr):
    v = [fmt(x) for x in np.asarray(arr).flatten()]
    f.write(f"static const float16 {name}[] = {{\n")
    for i in range(0, len(v), 8):
        f.write("    " + ", ".join(v[i : i + 8]) + ",\n")
    f.write("};\n\n")


def open_data(defines):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.h")
    f = open(path, "w")
    f.write("// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.\n")
    f.write("// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n")
    f.write("// SPDX-License-Identifier: Apache-2.0\n\n")
    f.write("/* Auto-generated by generator.py -- do not edit by hand.\n")
    f.write(" * Regenerate with: python3 generator.py\n */\n\n")
    f.write("#ifndef DATA_H_\n#define DATA_H_\n\n")
    for k, v in defines.items():
        f.write(f"#define {k} ({v})\n")
    f.write("\n/* clang-format off */\n")
    return f, path


def close_data(f):
    f.write("/* clang-format on */\n\n#endif /* DATA_H_ */\n")
    f.close()


# --- main --------------------------------------------------------------------


def main():
    rng = np.random.default_rng(SEED)

    X = rng.standard_normal((IN_N, IN_C, IN_H, IN_W)).astype(np.float16)

    filters = {}
    for name, out_c, groups in FILTERS:
        W = rng.standard_normal((out_c, IN_C // groups, KERNEL_H, KERNEL_W)).astype(np.float16)
        Bs = rng.standard_normal(out_c).astype(np.float16)
        filters[name] = (W, Bs, out_c, groups)

    defines = {
        "IN_N": IN_N,
        "IN_C": IN_C,
        "IN_H": IN_H,
        "IN_W": IN_W,
        "KERNEL_H": KERNEL_H,
        "KERNEL_W": KERNEL_W,
        "HAS_BIAS": HAS_BIAS,
        "NUM_CASES": len(CASES),
        "ULP_TOLL": 0,
    }

    goldens = []

    for tag, stride, pad, fname in CASES:
        W, Bs, out_c, groups = filters[fname]

        h_out = out_dim(IN_H, KERNEL_H, stride, pad)
        w_out = out_dim(IN_W, KERNEL_W, stride, pad)

        Y = conv_fp16_exact(X[0], W, Bs if HAS_BIAS else None, groups, stride, pad, h_out, w_out)

        defines[f"STRIDE_{tag}"] = stride
        defines[f"PAD_{tag}"] = pad
        defines[f"GROUPS_{tag}"] = groups
        defines[f"OUT_C_{tag}"] = out_c
        defines[f"OUT_H_{tag}"] = h_out
        defines[f"OUT_W_{tag}"] = w_out
        defines[f"OUT_LEN_{tag}"] = Y.size

        goldens.append((tag, Y))
        print(
            f"case {tag}: stride {stride} pad {pad} group {groups} -> out"
            f" {out_c}x{h_out}x{w_out} ({Y.size} elems),"
            f" |Y|max = {float(np.abs(Y.astype(np.float64)).max()):.4g}"
        )

    defines["OUT_LEN_MAX"] = max(int(y.size) for _, y in goldens)

    f, path = open_data(defines)
    emit(f, "X", X)
    for name, _, _ in FILTERS:
        W, Bs, _, _ = filters[name]
        emit(f, f"WEIGHTS{name}", W)
        emit(f, f"BIASES{name}", Bs)
    for tag, Y in goldens:
        emit(f, f"G_{tag}", Y)
    close_data(f)

    print(f"wrote {path}")


if __name__ == "__main__":
    main()
