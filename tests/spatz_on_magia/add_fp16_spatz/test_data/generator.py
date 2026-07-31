# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Input data and golden model for the FP16 Spatz add kernel test.

NumPy widens the two FP16 operands to FP32 (exact), adds and rounds once to FP16,
which is bit-identical to an IEEE-754 half-precision add with round-to-nearest-even -
so the test requires bit-exact results, including from the two broadcast forms, which
do the same single add per element and are therefore just NumPy broadcasting.

Three cases, one per entry point:

  E) elementwise, MAGIA_add_fp16_spatz over a flat element range.
  R) ADD_BCAST_ROW: A is [rows, row_len], B one row_len row reused by every row - the
     bias of a MatMul, and the attention bias shared by every window. The task reloads
     B per row (vfadd.vv).
  S) ADD_BCAST_SCALAR: B is one scalar per row. The task broadcasts it with vfadd.vf.

The element count is deliberately neither a multiple of the tile count (1, 4, 16, 64)
nor of Spatz's VLMAX for e16/m8 (256 elements @ VLEN=512), so both the uneven shard
split and the vector loop tail are exercised. The broadcast cases use 27 rows, which
divides evenly into neither mesh size, and an even row_len so the vector path rather
than the scalar fallback runs (rows are row_len apart in L1, so an odd one would
misalign every other row).
"""

import os

import numpy as np

LEN = 902

ROWS = 27
ROW_LEN = 34

# Operands stay well inside the FP16 range, so no result can overflow to Inf.
VALUE_RANGE = 4.0


def fmt(x):
    """'%.9g' of the exact value of an FP16 always round-trips through float, so the
    constants in the header are the very values the golden model used."""
    text = f"{float(x):.9g}"

    if "." not in text and "e" not in text and "n" not in text:
        text += ".0"

    return text + "f"


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


def main():
    rng = np.random.default_rng(0)

    A = rng.uniform(-VALUE_RANGE, VALUE_RANGE, LEN).astype(np.float16)
    B = rng.uniform(-VALUE_RANGE, VALUE_RANGE, LEN).astype(np.float16)
    G = (A + B).astype(np.float16)

    A2 = rng.uniform(-VALUE_RANGE, VALUE_RANGE, (ROWS, ROW_LEN)).astype(np.float16)
    B_row = rng.uniform(-VALUE_RANGE, VALUE_RANGE, ROW_LEN).astype(np.float16)
    B_scalar = rng.uniform(-VALUE_RANGE, VALUE_RANGE, ROWS).astype(np.float16)
    G_row = (A2 + B_row[None, :]).astype(np.float16)
    G_scalar = (A2 + B_scalar[:, None]).astype(np.float16)

    f = open_data({
        "VEC_LEN": LEN, "OUT_LEN_E": LEN,
        "ROWS": ROWS, "ROW_LEN": ROW_LEN, "OUT_LEN_B": A2.size,
        "OUT_LEN_MAX": max(LEN, A2.size),
        "NUM_CASES": 3, "ULP_TOLL": 0,
    })
    emit(f, "A", A)
    emit(f, "B", B)
    emit(f, "G", G)
    emit(f, "A2", A2)
    emit(f, "B_ROW", B_row)
    emit(f, "B_SCALAR", B_scalar)
    emit(f, "G_ROW", G_row)
    emit(f, "G_SCALAR", G_scalar)
    close_data(f)
    print(f"data.h generated [elementwise {LEN}, broadcast {ROWS}x{ROW_LEN}]")


if __name__ == "__main__":
    main()
