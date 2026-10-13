# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Input data and golden model for the FP16 Spatz transpose kernel test.

Pure data movement, so the results must be exact.

The kernel shards whichever of the input's axis 0 and the output's axis 0 is longer
and moves the side that is then non-contiguous with a 2-D transfer, so the cases below
have to cover both choices - and, for the input-axis-0 choice, both a perm that leaves
axis 0 in place (everything contiguous) and one that does not (rectangle out). Anything
that only ever ran perm[0] == 0 would miss the whole strided path.

The innermost (contiguous) dimension is deliberately ODD in several cases. The Spatz
VLSU corrupts vector stores to addresses that are not 4-byte aligned, and an odd row
length puts every other output row on a 2-byte address - so those shapes exercise the
task's alignment guard, which falls back to a scalar copy for such rows. Before that
guard was enabled the first configuration failed on ~40% of the elements.
"""

import os

import numpy as np

MAX_RANK = 4

# (in_shape, perm, what it exercises)
CASES = [
    ((18, 5, 7),  (0, 2, 1), "perm[0] == 0: contiguous in and out, odd inner"),
    ((12, 10, 3), (1, 0, 2), "shards in axis 0, q == 1: rectangle out, odd inner"),
    ((5, 14, 4),  (1, 0, 2), "shards out axis 0: rectangle in, contiguous out"),
    ((6, 7, 5),   (2, 0, 1), "shards in axis 0, q == 1, inner from a moved axis"),
    ((19, 28),    (1, 0),    "rank 2, shards out axis 0: a plain matrix transpose"),
    ((3, 40, 2),  (1, 2, 0), "shards out axis 0, in axis 0 last: both sides strided"),
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


def emit_u32_table(f, name, rows, width):
    f.write(f"static const uint32_t {name}[][{width}] = {{\n")
    for r in rows:
        padded = list(r) + [0] * (width - len(r))
        f.write("    {" + ", ".join(str(int(v)) for v in padded) + "},\n")
    f.write("};\n\n")


def emit_u32_list(f, name, vals):
    f.write(f"static const uint32_t {name}[] = {{")
    f.write(", ".join(str(int(v)) for v in vals))
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
    f.write("#include <stdint.h>\n\n")
    for k, v in defines.items():
        f.write(f"#define {k:<12} {v}\n")
    f.write("\n/* clang-format off */\n")
    return f


def close_data(f):
    f.write("/* clang-format on */\n\n#endif /* DATA_H_ */\n")
    f.close()


def main():
    rng = np.random.default_rng(0)

    xs = []
    gs = []
    ranks = []
    in_shapes = []
    out_shapes = []
    perms = []
    lens = []
    offs = []

    off = 0
    for in_shape, perm, _ in CASES:
        out_shape = tuple(in_shape[p] for p in perm)

        X = rng.uniform(-4.0, 4.0, in_shape).astype(np.float16)
        G = np.transpose(X, perm).copy()

        xs.append(X.reshape(-1))
        gs.append(G.reshape(-1))
        ranks.append(len(in_shape))
        in_shapes.append(in_shape)
        out_shapes.append(out_shape)
        perms.append(perm)
        lens.append(X.size)
        offs.append(off)
        off += X.size

    f = open_data({"NUM_CASES": len(CASES),
                   "MAX_RANK": MAX_RANK,
                   "MAX_LEN": max(lens),
                   "ULP_TOLL": 0})

    emit_u32_list(f, "CASE_RANK", ranks)
    emit_u32_list(f, "CASE_LEN", lens)
    emit_u32_list(f, "CASE_OFF", offs)
    emit_u32_table(f, "CASE_IN_SHAPE", in_shapes, MAX_RANK)
    emit_u32_table(f, "CASE_OUT_SHAPE", out_shapes, MAX_RANK)
    emit_u32_table(f, "CASE_PERM", perms, MAX_RANK)

    emit(f, "X", np.concatenate(xs))
    emit(f, "G", np.concatenate(gs))
    close_data(f)

    for (in_shape, perm, why), n in zip(CASES, lens):
        out_shape = tuple(in_shape[p] for p in perm)
        print(f"  {in_shape} perm {perm} -> {out_shape}  ({n} elements) - {why}")
    print(f"data.h generated [{len(CASES)} cases, {off} elements]")


if __name__ == "__main__":
    main()
