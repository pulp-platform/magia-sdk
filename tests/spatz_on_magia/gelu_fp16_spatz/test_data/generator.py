# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Input data and golden model for the FP16 Spatz gelu kernel test.

The kernel evaluates the tanh form of GELU,

    0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))

and approximates the tanh with a Schraudolph-style fast exp: with
e ~ bitcast_fp16(uint(2 * COEF * t + BIAS)) standing in for exp(2t), it returns
(e - 1) / (e + 1). The golden model replays exactly that sequence in FP16 - every
intermediate rounding, and the truncating float-to-unsigned conversion - so the test
is bit-exact and pins the approximation rather than tolerating it. Both deviations
(from the tanh form, and from the exact-erf GELU that ONNX's Gelu op means with
approximate="none") are reported when regenerating; they are properties of the kernel,
not errors.

Unlike sigmoid, the fast exp here cannot be pushed out of range: the kernel clamps its
argument to +/-5 with vfmin/vfmax *before* the exp, so 2 * COEF * t + BIAS stays inside
[500, 30220] whatever the input. The clamp is also what rescues the x^3 intermediate,
which overflows FP16 above |x| ~ 40.3 - the resulting +/-Inf clamps straight to +/-5.
The SENTINELS below straddle the clamp threshold (|x| ~ 3.6), the x^3 overflow, and go
well past both.

The element count is deliberately neither a multiple of the tile count (1, 4, 16, 64)
nor of Spatz's VLMAX for e16/m8 (256 elements @ VLEN=512), so both the uneven shard
split and the vector loop tail are exercised.
"""

import os

import numpy as np

LEN = 902

C0   = np.float16(0.044715)
C1   = np.float16(0.797884561)   # sqrt(2/pi)
TMIN = np.float16(-5.0)
TMAX = np.float16(5.0)
COEF = np.float16(1486.0)
BIAS = np.float16(15360.0)
ONE  = np.float16(1.0)
TWO  = np.float16(2.0)
HALF = np.float16(0.5)

# Straddle the clamp threshold (|x| ~ 3.6) and the x^3 overflow (|x| ~ 40.3), and go
# well past both. Placed at the front of X, which is hart 0's shard at every mesh size.
SENTINELS = [-1000.0, -100.0, -40.5, -40.0, -10.0, -4.0, -3.6, -3.0, -1.0, -0.5,
             0.0, 0.5, 1.0, 3.0, 3.6, 4.0, 10.0, 40.0, 40.5, 100.0, 1000.0]


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


def gelu_kernel_model(X):
    """vfmul.vv x3 / vfadd.vv / vfmul.vf / vfmin.vf / vfmax.vf / the fast tanh / vfmul.

    x^3 overflows to +/-Inf above |x| ~ 40.3; numpy raises on that, and it is exactly
    what the hardware does too, so it is silenced. The clamp two steps later turns the
    Inf back into +/-5.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        X = X.astype(np.float16)

        t = (X * X).astype(np.float16)
        t = (t * X).astype(np.float16)
        t = (t * C0).astype(np.float16)
        t = (t + X).astype(np.float16)
        t = (t * C1).astype(np.float16)

        t = np.minimum(t, TMAX).astype(np.float16)
        t = np.maximum(t, TMIN).astype(np.float16)

        u = (t * TWO).astype(np.float16)
        u = (u * COEF).astype(np.float16)
        u = (u + BIAS).astype(np.float16)

        # vfcvt.rtz.xu.f.v: truncate towards zero into an unsigned 16-bit integer,
        # saturating, then reinterpret those bits as FP16.
        ui = np.clip(np.trunc(u.astype(np.float64)), 0, 65535).astype(np.uint16)
        e = ui.view(np.float16)

        den = (e + ONE).astype(np.float16)
        num = (e - ONE).astype(np.float16)
        th = (num / den).astype(np.float16)

        r = (th + ONE).astype(np.float16)
        r = (r * X).astype(np.float16)
        return (r * HALF).astype(np.float16)


def main():
    rng = np.random.default_rng(0)

    X = rng.uniform(-6.0, 6.0, LEN).astype(np.float16)
    X[: len(SENTINELS)] = np.asarray(SENTINELS, dtype=np.float16)

    G = gelu_kernel_model(X)

    x64 = X.astype(np.float64)
    # The function the kernel is approximating ...
    inner = np.sqrt(2.0 / np.pi) * (x64 + 0.044715 * x64 ** 3)
    tanh_gelu = 0.5 * x64 * (1.0 + np.tanh(inner))
    # ... and the one ONNX's Gelu(approximate="none") actually means.
    from math import erf
    erf_gelu = np.array([0.5 * v * (1.0 + erf(v / np.sqrt(2.0))) for v in x64])

    g64 = G.astype(np.float64)
    finite = np.isfinite(g64) & np.isfinite(tanh_gelu) & np.isfinite(erf_gelu)
    dev_tanh = np.max(np.abs(g64[finite] - tanh_gelu[finite]))
    dev_erf = np.max(np.abs(g64[finite] - erf_gelu[finite]))

    f = open_data({"VEC_LEN": LEN, "OUT_LEN": LEN, "ULP_TOLL": 0})
    emit(f, "X", X)
    emit(f, "G", G)
    close_data(f)

    print(f"data.h generated [LEN:{LEN}]")
    print(f"kernel deviates from the tanh GELU by up to {dev_tanh:.4f}, "
          f"and from the exact-erf GELU by up to {dev_erf:.4f}")


if __name__ == "__main__":
    main()
