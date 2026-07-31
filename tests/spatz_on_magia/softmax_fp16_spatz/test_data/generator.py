# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Input data and golden model for the FP16 Spatz softmax kernel test.

Softmax over the last axis. Like the sigmoid kernel this one uses a
Schraudolph-style fast exp - here with COEF = 1486 - so the golden model replays
the kernel's sequence in FP16: row max, exp approximation of (x - max), sum,
then a per-element FP16 divide, and the whole thing is bit-exact: the lane fold
is vfredosum, so RVV pins its order down (it used to be vfredusum, which does
not, and the test carried a ULP budget to cover it). The deviation from a true
softmax is reported when regenerating and is a property of the kernel.

Two cases:

  A) 1x2x9x17 - the row length is deliberately ODD. The Spatz VLSU corrupts
     vector accesses to addresses that are not 4-byte aligned, and an odd row
     length puts every other row on a 2-byte address, so this exercises the
     kernel's alignment guard and its scalar fallback. Without that guard this
     configuration is wrong on a large fraction of the elements.

  B) 1x2x9x300 - even, so the vector path runs, and longer than VLMAX, so the
     sum goes through the lane accumulator before the fold. That is the only
     part of the kernel that can tell VLMAX = 256 from VLMAX = 128, and getting
     it wrong is silently invisible in any row that fits a single chunk.
"""

import os

import numpy as np

# VLMAX at e16/m8 is 256, not 128: the magia_v3 GVSoC model gives Spatz VLEN = 512,
# whatever cmake/spatz_config.cmake's (unused) SPATZ_VLEN says.
VLMAX = 256

# tag, N, C, H, W
CASES = [
    ("A", 1, 2, 9, 17),
    ("B", 1, 2, 9, 300),
]

COEF = np.float16(1486.0)
BIAS = np.float16(15360.0)

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
        f.write(f"#define {k:<14} {v}\n")
    f.write("\n/* clang-format off */\n")
    return f


def close_data(f):
    f.write("/* clang-format on */\n\n#endif /* DATA_H_ */\n")
    f.close()


def approx_exp(v):
    """vfmul.vf / vfadd.vf / vfcvt.rtz.xu.f.v, result reinterpreted as FP16"""
    t = (v * COEF).astype(np.float16)
    t = (t + BIAS).astype(np.float16)
    u = np.clip(np.trunc(t.astype(np.float64)), 0, 65535).astype(np.uint16)
    return u.view(np.float16)


def row_sum(e):
    """Whole VLMAX-wide chunks added element-wise into a lane accumulator (vfadd.vv,
    ascending), then the lanes folded ascending from zero (vfredosum.vs). A row that
    fits one chunk reduces to a plain ascending sum."""
    lanes = np.zeros(min(VLMAX, e.size), dtype=np.float16)

    for off in range(0, e.size, VLMAX):
        chunk = e[off : off + VLMAX]
        lanes[: chunk.size] = (
            lanes[: chunk.size].astype(np.float64) + chunk.astype(np.float64)
        ).astype(np.float16)

    acc = np.float16(0.0)
    for v in lanes:
        acc = (acc + v).astype(np.float16)

    return acc


def main():
    rng = np.random.default_rng(0)

    defines = {"ULP_TOLL": 0, "NUM_CASES": len(CASES)}
    arrays = []
    out_lens = []

    for tag, n, c, h, w in CASES:
        X = rng.uniform(-4.0, 4.0, (n, c, h, w)).astype(np.float16)

        rows = X.reshape(-1, w)
        G = np.zeros_like(rows)
        for r in range(rows.shape[0]):
            row = rows[r]
            e = approx_exp((row - row.max()).astype(np.float16))
            G[r] = (e / row_sum(e)).astype(np.float16)

        r64 = rows.astype(np.float64)
        true_sm = np.exp(r64 - r64.max(axis=1, keepdims=True))
        true_sm /= true_sm.sum(axis=1, keepdims=True)
        max_dev = np.max(np.abs(G.astype(np.float64) - true_sm))

        defines[f"IN_N_{tag}"] = n
        defines[f"IN_C_{tag}"] = c
        defines[f"IN_H_{tag}"] = h
        defines[f"IN_W_{tag}"] = w
        defines[f"OUT_LEN_{tag}"] = X.size

        arrays += [(f"X_{tag}", X), (f"G_{tag}", G.reshape(X.shape))]
        out_lens.append(int(X.size))

        print(f"case {tag}: {n}x{c}x{h}x{w} ({w} per row, {-(-w // VLMAX)} vector chunk(s)) -"
              f" deviation from true softmax up to {max_dev:.4f}")

    defines["OUT_LEN_MAX"] = max(out_lens)

    f = open_data(defines)
    for name, arr in arrays:
        emit(f, name, arr)
    close_data(f)
    print("data.h generated")


if __name__ == "__main__":
    main()
