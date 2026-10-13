# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Input data and golden model for the FP16 Spatz groupnorm kernel test.

Normalizes over each group of channels (c_per_group * H * W elements), then
applies a per-channel scale and bias. n * num_groups is the sharded dimension.

Every step is real FP16 arithmetic, and the golden model replays it exactly -
including where the kernel deliberately leaves FP16 to stay in range. Two cases:

  S) 2x27x4x4, 9 groups - 18 shards, which divides evenly into neither 16 nor 64
     tiles, so the remainder shard is exercised. 48 elements per group.

  L) 1x64x32x32, 1 group - 65536 elements per group, the shape that made the
     original kernel unusable. Summing 65536 FP16 activations reaches ~1e5 and
     their squares ~1e6, both past FP16's 65504 ceiling, and (_Float16)65536 is
     itself Inf, so the old mean = sum/len came out 0 and the variance Inf. The
     kernel now takes both reductions in units of sh = the smallest power of two
     >= sqrt(len) - an exact scaling, since sh is a power of two - and finishes
     the two rescalings in FP32, where the variance is allowed to exceed the FP16
     range because only 1/sqrt(var + eps) is ever rounded back. S also fits in a
     single vector chunk where L needs 256 of them, so the two cases cover both
     sides of the accumulate-then-fold split - and only L can tell whether VLMAX
     below is right.

The lane folds are vfredosum, not vfredsum: RVV leaves the unordered fold's
association to the implementation, which no golden can model. That is why this is
bit-exact (ULP_TOLL 0) where it used to need a tolerance of 32.
"""

import os

import numpy as np

# VLMAX at e16/m8, i.e. how many elements share a lane accumulator. The magia_v3 GVSoC
# model gives its Spatz VLEN = 512 (gvsoc_work/gvsoc_config.json), so 512/16 * 8 = 256 -
# NOT the 128 that cmake/spatz_config.cmake's SPATZ_VLEN = 256 would suggest. That macro
# is not read by any kernel, so it does not make the hardware agree with it; vsetvli is
# the only thing that decides, and it says 256. Getting this wrong is invisible in any
# reduction short enough to fit one chunk and silently wrong in every longer one, which
# is exactly why case L below is 512 chunks long.
VLMAX = 256

EPS = 0.001

# tag, N, C, H, W, num_groups
CASES = [
    ("S", 2, 27, 4, 4, 9),
    ("L", 1, 64, 32, 32, 1),
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


def groupnorm_kernel_model(X, scale, bias, num_groups, eps16):
    n, c, h, w = X.shape
    c_per_g = c // num_groups
    npg = c_per_g * h * w

    sh = stat_scale(npg)
    inv_sh = np.float16(1.0 / sh)  # exact: sh is a power of two
    sh_over_n = np.float32(sh) / np.float32(npg)
    sh2_over_n = (np.float32(sh) * np.float32(sh)) / np.float32(npg)

    G = np.zeros_like(X)

    for b in range(n):
        for g in range(num_groups):
            block = X[b, g * c_per_g : (g + 1) * c_per_g].reshape(-1)

            # mean: sum of x/sh, rescaled in FP32
            s = lane_fold((block * inv_sh).astype(np.float16))
            mean = np.float16(np.float32(s) * sh_over_n)

            # variance: sum of ((x-mean)/sh)^2, rescaled in FP32 and left there
            d = (block - mean).astype(np.float16)
            ds = (d * inv_sh).astype(np.float16)
            ss = lane_fold((ds * ds).astype(np.float16))
            var = np.float32(ss) * sh2_over_n

            denom = np.float16(np.float32(1.0) / np.sqrt(var + np.float32(eps16)))

            norm = (d * denom).astype(np.float16).reshape(c_per_g, h, w)
            for ci in range(c_per_g):
                ch = g * c_per_g + ci
                G[b, ch] = ((norm[ci] * scale[ch]).astype(np.float16) + bias[ch]).astype(
                    np.float16
                )

    return G


def main():
    rng = np.random.default_rng(0)
    eps16 = np.float16(EPS)

    defines = {"EPSILON": f"{EPS}f", "ULP_TOLL": 0, "NUM_CASES": len(CASES)}
    arrays = []
    out_lens = []

    for tag, n, c, h, w, groups in CASES:
        X = rng.uniform(-3.0, 3.0, (n, c, h, w)).astype(np.float16)
        scale = rng.uniform(0.5, 1.5, c).astype(np.float16)
        bias = rng.uniform(-0.5, 0.5, c).astype(np.float16)

        G = groupnorm_kernel_model(X, scale, bias, groups, eps16)

        defines[f"IN_N_{tag}"] = n
        defines[f"IN_C_{tag}"] = c
        defines[f"IN_H_{tag}"] = h
        defines[f"IN_W_{tag}"] = w
        defines[f"NUM_GROUPS_{tag}"] = groups
        defines[f"OUT_LEN_{tag}"] = X.size

        arrays += [(f"X_{tag}", X), (f"SCALE_{tag}", scale), (f"BIAS_{tag}", bias), (f"G_{tag}", G)]
        out_lens.append(int(X.size))

        npg = (c // groups) * h * w
        flat = X.astype(np.float64).reshape(n * groups, npg)
        raw_sum = float(np.abs(flat.sum(axis=1)).max())
        raw_sq = float((flat**2).sum(axis=1).max())
        print(
            f"case {tag}: {n}x{c}x{h}x{w}, {groups} groups -> {npg} elems/group,"
            f" sh {stat_scale(npg)}; unscaled |sum| up to {raw_sum:.4g}, sum of squares"
            f" up to {raw_sq:.4g} ({'past' if raw_sq > 65504 else 'within'} FP16's 65504),"
            f" (float16){npg} = {np.float16(npg)}"
        )

    defines["OUT_LEN_MAX"] = max(out_lens)

    f = open_data(defines)
    for name, arr in arrays:
        emit(f, name, arr)
    close_data(f)
    print("data.h generated")


if __name__ == "__main__":
    main()
