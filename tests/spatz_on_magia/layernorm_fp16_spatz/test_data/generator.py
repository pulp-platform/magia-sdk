# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Input data and golden model for the FP16 Spatz layernorm kernel test.

Normalizes over the last axis, then applies a per-column scale and bias. Every step is
real FP16 arithmetic and the golden model replays it exactly - including where the
kernel deliberately leaves FP16 to stay in range.

The kernel takes both reductions in units of sh = the smallest power of two >= sqrt(len)
- an exact scaling, since sh is a power of two - and finishes the two rescalings in FP32,
where the variance is allowed to exceed the FP16 range because only 1/sqrt(var + eps) is
ever rounded back. That is the same treatment groupnorm needed, and for the same reason:
accumulating a long row's sum and sum of squares straight in FP16 runs out of range, and
(_Float16)len eventually becomes Inf on its own.

Both lane folds are vfredosum, not vfredsum. RVV leaves the unordered fold's association
to the implementation, which no golden can model; compute_mean used to fold with
vfredsum while compute_variance folded with vfredosum. That inconsistency is why this
test used to need a tolerance of 32 ULP and is bit-exact now.

Three cases:

  S) 18x20   - 18 rows, which divides evenly into neither 16 nor 64 tiles, so the
               remainder shard is exercised; one vector chunk per row.
  L) 7x320   - 320 > VLMAX, so each row needs two chunks and the lane accumulator is
               what decides the result. Only a case like this can tell whether VLMAX
               below is right. 320 is also TinyViT's longest LayerNorm row.
  W) 3x1024  - long rows of large activations, so that the raw sum of squares leaves
               FP16 range and only the scaling keeps the variance finite (reported
               below when regenerating).
"""

import os

import numpy as np

# VLMAX at e16/m8, i.e. how many elements share a lane accumulator. The magia_v3 GVSoC
# model gives its Spatz VLEN = 512 (gvsoc_work/gvsoc_config.json), so 512/16 * 8 = 256 -
# NOT the 128 that cmake/spatz_config.cmake's SPATZ_VLEN = 256 would suggest. That macro
# is not read by any kernel, so it does not make the hardware agree with it; vsetvli is
# the only thing that decides, and it says 256.
VLMAX = 256

EPS = 0.001

# tag, rows, row_len, |x| range
CASES = [
    ("S", 18, 20, 3.0),
    ("L", 7, 320, 3.0),
    ("W", 3, 1024, 30.0),
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
        f.write(f"#define {k:<16} {v}\n")
    f.write("\n/* clang-format off */\n")
    return f


def close_data(f):
    f.write("/* clang-format on */\n\n#endif /* DATA_H_ */\n")
    f.close()


def stat_scale(n):
    """Smallest power of two >= sqrt(n), the unit both reductions are taken in."""
    sh = 1
    while sh < 32768 and sh * sh < n:
        sh <<= 1
    return sh


def lane_fold(vals):
    """Whole VLMAX-wide chunks added element-wise into a lane accumulator
    (vfadd.vv, ascending chunk order), then the lanes folded ascending
    (vfredosum.vs, starting from zero)."""
    lanes = np.zeros(min(VLMAX, vals.size), dtype=np.float16)

    for off in range(0, vals.size, VLMAX):
        chunk = vals[off : off + VLMAX]
        lanes[: chunk.size] = (
            lanes[: chunk.size].astype(np.float64) + chunk.astype(np.float64)
        ).astype(np.float16)

    acc = np.float16(0.0)
    for v in lanes:
        acc = (acc + v).astype(np.float16)

    return acc


def layernorm_kernel_model(X, scale, bias, eps16):
    rows, w_len = X.shape

    sh = stat_scale(w_len)
    inv_sh = np.float16(1.0 / sh)  # exact: sh is a power of two
    sh_over_n = np.float32(sh) / np.float32(w_len)
    sh2_over_n = (np.float32(sh) * np.float32(sh)) / np.float32(w_len)

    G = np.zeros_like(X)

    for r in range(rows):
        row = X[r]

        # mean: sum of x/sh, rescaled in FP32
        s = lane_fold((row * inv_sh).astype(np.float16))
        mean = np.float16(np.float32(s) * sh_over_n)

        # variance: sum of ((x-mean)/sh)^2, rescaled in FP32 and left there
        d = (row - mean).astype(np.float16)
        ds = (d * inv_sh).astype(np.float16)
        ss = lane_fold((ds * ds).astype(np.float16))
        var = np.float32(ss) * sh2_over_n

        denom = np.float16(np.float32(1.0) / np.sqrt(var + np.float32(eps16)))

        norm = (d * denom).astype(np.float16)
        G[r] = ((norm * scale).astype(np.float16) + bias).astype(np.float16)

    return G


def main():
    rng = np.random.default_rng(0)
    eps16 = np.float16(EPS)

    defines = {"EPSILON": f"{EPS}f", "ULP_TOLL": 0, "NUM_CASES": len(CASES), "RANK": 2}
    arrays = []
    out_lens = []

    for tag, rows, w_len, span in CASES:
        X = rng.uniform(-span, span, (rows, w_len)).astype(np.float16)
        scale = rng.uniform(0.5, 1.5, w_len).astype(np.float16)
        bias = rng.uniform(-0.5, 0.5, w_len).astype(np.float16)

        G = layernorm_kernel_model(X, scale, bias, eps16)

        defines[f"DIM_0_{tag}"] = rows
        defines[f"DIM_1_{tag}"] = w_len
        defines[f"OUT_LEN_{tag}"] = X.size

        arrays += [(f"X_{tag}", X), (f"SCALE_{tag}", scale), (f"BIAS_{tag}", bias), (f"G_{tag}", G)]
        out_lens.append(int(X.size))

        flat = X.astype(np.float64)
        raw_sum = float(np.abs(flat.sum(axis=1)).max())
        raw_sq = float((flat**2).sum(axis=1).max())
        print(
            f"case {tag}: {rows}x{w_len}, sh {stat_scale(w_len)}, "
            f"{-(-w_len // VLMAX)} vector chunk(s) per row; unscaled |sum| up to "
            f"{raw_sum:.4g}, sum of squares up to {raw_sq:.4g} "
            f"({'past' if raw_sq > 65504 else 'within'} FP16's 65504)"
        )

    defines["OUT_LEN_MAX"] = max(out_lens)

    f = open_data(defines)
    for name, arr in arrays:
        emit(f, name, arr)
    close_data(f)
    print("data.h generated")


if __name__ == "__main__":
    main()
