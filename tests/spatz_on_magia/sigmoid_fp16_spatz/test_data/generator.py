# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Input data and golden model for the FP16 Spatz sigmoid kernel test.

The kernel does not evaluate a true sigmoid: it uses a Schraudolph-style fast
exp, i.e. exp(-x) ~ bitcast_fp16(uint(COEF * -x + BIAS)) with COEF = 2^10/ln2
and BIAS = 15 * 2^10, then 1 / (1 + that). The golden model replays exactly that
sequence in FP16 - including each intermediate rounding and the truncating
float-to-unsigned conversion - so the test can be bit-exact and pins the
approximation instead of tolerating it. The deviation from a true sigmoid is
reported below when regenerating, and is a property of the kernel, not an error.

That fast exp is only usable over a limited input range, and the inputs here are
chosen to straddle both ends of it. COEF * -x + BIAS has to stay a positive
finite FP16: beyond -x = 11.09 it crosses 32768 and the reinterpreted "exp"
comes back negative, and beyond -x = 34 the truncating conversion saturates to
0xFFFF, which is an FP16 NaN. So the kernel clamps x at MIN = -11 first, which
caps the result at sigmoid(-11) = 1.7e-5 - an absolute error two orders below
the fast exp's own deviation. The SENTINELS below sit on both sides of both
thresholds; without the clamp the last of them turn the output into NaN, which
is what mobilevit_v2 (sigmoid inputs down to -898) ran into.
"""

import os

import numpy as np

LEN  = 902
COEF = np.float16(1477.0)
BIAS = np.float16(15360.0)
MIN  = np.float16(-11.0)

# Straddle the sign-flip threshold (-11.09) and the NaN threshold (-34), and go well
# past both. Placed at the front of X, which is hart 0's shard at every mesh size.
SENTINELS = [-1000.0, -898.0, -64.0, -34.5, -34.0, -33.0, -11.5, -11.09, -11.0,
             -10.9, -8.0, 0.0, 8.0, 900.0]

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


def sigmoid_kernel_model(X, clamp=True):
    """vfmax.vf / vfsgnjn / vfmul.vf / vfadd.vf / vfcvt.rtz.xu.f.v / vfadd.vf / vfdiv.vv

    The clamp only bounds x from below, so a large positive x still overflows
    COEF * -x to -Inf. That is the harmless direction: the conversion saturates a
    negative operand to 0, exp comes out 0 and the result is 1, which is the right
    answer. numpy raises on those overflows all the same, so they are silenced -
    as are the ones the clamp=False model raises on purpose.
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if clamp:
            X = np.maximum(X.astype(np.float16), MIN).astype(np.float16)
        neg = (-X).astype(np.float16)
        t = (neg * COEF).astype(np.float16)
        t = (t + BIAS).astype(np.float16)
        u = np.trunc(t.astype(np.float64))
        u = np.clip(u, 0, 65535).astype(np.uint16)
        approx_exp = u.view(np.float16)
        den = (np.float16(1.0) + approx_exp).astype(np.float16)
        return (np.float16(1.0) / den).astype(np.float16)


def main():
    rng = np.random.default_rng(0)

    X = rng.uniform(-40.0, 40.0, LEN).astype(np.float16)
    X[: len(SENTINELS)] = np.asarray(SENTINELS, dtype=np.float16)

    G = sigmoid_kernel_model(X)

    # Only where the clamp is not what decides the answer, i.e. where the fast exp is
    # actually being measured against the function it approximates.
    live = X >= MIN
    true_sigmoid = 1.0 / (1.0 + np.exp(-X[live].astype(np.float64)))
    max_dev = np.max(np.abs(G[live].astype(np.float64) - true_sigmoid))

    # What the kernel did before the clamp existed.
    unclamped = sigmoid_kernel_model(X, clamp=False).astype(np.float64)
    n_nan = int(np.isnan(unclamped).sum())
    n_bad = int(np.sum(~np.isfinite(unclamped) | (unclamped < 0)))

    f = open_data({"VEC_LEN": LEN, "OUT_LEN": LEN, "ULP_TOLL": 0})
    emit(f, "X", X)
    emit(f, "G", G)
    close_data(f)
    print(f"data.h generated [LEN:{LEN}] - kernel deviates from true sigmoid by up to "
          f"{max_dev:.4f} over the {int(live.sum())} unclamped inputs")
    print(f"without the clamp, {n_nan} of the {LEN} inputs would give NaN "
          f"and {n_bad} a NaN or a negative 'probability'")


if __name__ == "__main__":
    main()
