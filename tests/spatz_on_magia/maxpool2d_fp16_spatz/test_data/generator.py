# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Input data and golden model for the FP16 Spatz maxpool2d kernel test.

A maximum selects an input value unchanged, so the results must be exact.

Two configurations are emitted, because the kernel takes a different path for each:

  A) 2x2, stride 2, pad 0 - every window starts on a 4-byte boundary, so the vector
     path runs.

  B) 3x3, stride 2, pad 1 - ResNet18's pool. A window starts at win_w_start = 2*ow - 1
     clamped to 0, which is odd from ow = 1 on, and the Spatz VLSU corrupts vector
     accesses that are not 4-byte aligned. The kernel detects that and falls back to a
     scalar window max, so this case is what covers the guard.

The kernel clamps each window to the input rather than padding it, which is what ONNX
MaxPool's -inf padding amounts to, so the golden clamps too.
"""

import os

import numpy as np

IN_N = 2
IN_C = 9
IN_H = 8
IN_W = 8

# (tag, kernel, stride, pad): A is the aligned vector path, B the ResNet18-shaped
# padded pool that exercises the misaligned-window fallback.
CASES = [
    ("A", 2, 2, 0),
    ("B", 3, 2, 1),
]

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
    X = rng.uniform(-4.0, 4.0, (IN_N, IN_C, IN_H, IN_W)).astype(np.float16)

    defines = {"IN_N": IN_N, "IN_C": IN_C, "IN_H": IN_H, "IN_W": IN_W, "ULP_TOLL": 0}
    goldens = []

    for tag, kernel, stride, pad in CASES:
        out_h = (IN_H + 2 * pad - kernel) // stride + 1
        out_w = (IN_W + 2 * pad - kernel) // stride + 1
        G = np.zeros((IN_N, IN_C, out_h, out_w), dtype=np.float16)

        for n in range(IN_N):
            for c in range(IN_C):
                for oh in range(out_h):
                    # The window is clamped to the input, not padded: ONNX pads with
                    # -inf, which for a maximum is the same thing.
                    h0 = max(oh * stride - pad, 0)
                    h1 = min(oh * stride - pad + kernel, IN_H)
                    for ow in range(out_w):
                        w0 = max(ow * stride - pad, 0)
                        w1 = min(ow * stride - pad + kernel, IN_W)
                        G[n, c, oh, ow] = X[n, c, h0:h1, w0:w1].max()

        defines[f"KERNEL_{tag}"] = kernel
        defines[f"STRIDE_{tag}"] = stride
        defines[f"PAD_{tag}"] = pad
        defines[f"OUT_H_{tag}"] = out_h
        defines[f"OUT_W_{tag}"] = out_w
        defines[f"OUT_LEN_{tag}"] = G.size

        goldens.append((tag, G))
        print(f"case {tag}: {kernel}x{kernel} stride {stride} pad {pad}"
              f" -> {out_h}x{out_w} ({G.size} elems)")

    defines["OUT_LEN_MAX"] = max(int(g.size) for _, g in goldens)

    f = open_data(defines)
    emit(f, "X", X)
    for tag, G in goldens:
        emit(f, f"G_{tag}", G)
    close_data(f)

    print(f"data.h generated [{IN_N}x{IN_C}x{IN_H}x{IN_W}]")


if __name__ == "__main__":
    main()
