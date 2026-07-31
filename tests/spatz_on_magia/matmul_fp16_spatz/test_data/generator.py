# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Input data and golden model for the FP16 Spatz matmul kernel test.

Y[b] = A[b] @ B[b]. The kernel accumulates with vfmacc, a fused multiply-add over
ascending k, and its scalar fallback does the same with fmadd.h - one rounding per
term either way - so the golden model replays exactly that and the results must be
bit-exact.

Four cases, covering what the kernel actually has to get right:

  0) batched, even O          - the vector path, several batches
  1) batched, ODD O           - the alignment guard and its scalar fallback
  2) shared B (b_batched = 0)  - one B reused by every batch, staged once
  3) a single batch            - the M sharding is all that spreads this one

Case 1 is the important one. The Spatz VLSU corrupts vector accesses to addresses
that are not 4-byte aligned, and the B rows and the Y rows both step by O, so an odd
O puts every other row on a 2-byte address. It is the shape of TinyViT's 7x7 QK^T
scores ([49,32] @ [32,49]), and without the guard this configuration is wrong on a
large fraction of the elements.

M is kept uneven over both mesh sizes (16 and 64 tiles) so the shard split has a
remainder, and small enough that 64 tiles leaves most of them with nothing to do -
which is what exercises the empty-shard guard.
"""

import os

import numpy as np

# (M, K, O, batches, a_batched, b_batched, what it covers)
CASES = [
    (13, 7, 8, 5, 1, 1, "batched, even O: the vector path"),
    (9, 5, 7, 3, 1, 1, "batched, ODD O: the alignment guard and scalar fallback"),
    (11, 6, 10, 4, 1, 0, "shared B, staged once and reused by every batch"),
    (17, 9, 12, 1, 1, 1, "a single batch: only the M sharding spreads it"),
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


def fma16(acc, a, b):
    """One FP16 fused multiply-add: exact product, single rounding (vfmacc / fmadd.h)"""
    return np.float16(float(acc) + float(a) * float(b))


def matmul_fp16_exact(A, B, M, K, O):
    Y = np.zeros((M, O), dtype=np.float16)

    for m in range(M):
        for o in range(O):
            acc = np.float16(0.0)
            for k in range(K):              # vfmacc, ascending k
                acc = fma16(acc, A[m, k], B[k, o])
            Y[m, o] = acc

    return Y


def main():
    rng = np.random.default_rng(0)

    a_all, b_all, g_all = [], [], []
    a_off, b_off, y_off = [], [], []
    y_len = []
    dims = {"M": [], "K": [], "O": [], "BATCHES": [], "A_BATCHED": [], "B_BATCHED": []}

    ao = bo = yo = 0
    for M, K, O, batches, a_batched, b_batched in [c[:6] for c in CASES]:
        a_batches = batches if a_batched else 1
        b_batches = batches if b_batched else 1

        A = rng.uniform(-1.5, 1.5, (a_batches, M, K)).astype(np.float16)
        B = rng.uniform(-1.5, 1.5, (b_batches, K, O)).astype(np.float16)

        G = np.stack([matmul_fp16_exact(A[b if a_batched else 0],
                                        B[b if b_batched else 0], M, K, O)
                      for b in range(batches)])

        a_all.append(A.reshape(-1))
        b_all.append(B.reshape(-1))
        g_all.append(G.reshape(-1))

        a_off.append(ao)
        b_off.append(bo)
        y_off.append(yo)
        y_len.append(G.size)

        ao += A.size
        bo += B.size
        yo += G.size

        dims["M"].append(M)
        dims["K"].append(K)
        dims["O"].append(O)
        dims["BATCHES"].append(batches)
        dims["A_BATCHED"].append(a_batched)
        dims["B_BATCHED"].append(b_batched)

    f = open_data({"NUM_CASES": len(CASES), "MAX_LEN": max(y_len), "ULP_TOLL": 0})

    for name, vals in dims.items():
        emit_u32_list(f, f"CASE_{name}", vals)
    emit_u32_list(f, "CASE_A_OFF", a_off)
    emit_u32_list(f, "CASE_B_OFF", b_off)
    emit_u32_list(f, "CASE_Y_OFF", y_off)
    emit_u32_list(f, "CASE_Y_LEN", y_len)

    emit(f, "A", np.concatenate(a_all))
    emit(f, "B", np.concatenate(b_all))
    emit(f, "G", np.concatenate(g_all))
    close_data(f)

    for (M, K, O, batches, a_batched, b_batched, why), n in zip(CASES, y_len):
        print(f"  [{batches if a_batched else 1}]({M}x{K}) @ [{batches if b_batched else 1}]"
              f"({K}x{O}) -> {n} elements - {why}")
    print(f"data.h generated [{len(CASES)} cases]")


if __name__ == "__main__":
    main()
