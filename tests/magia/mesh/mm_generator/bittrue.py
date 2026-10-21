# Copyright 2023-2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Yvan Tortorella <yvan.tortorella@unibo.it>
# Francesco Conti <francesco.conti@chips.it>
#
# Bit-true FP16 helpers shared by the golden-data generators.

import numpy as np


def bittrue_fma(a, x, b):
    # Simulate FP16 FMA with intermediate rounding
    a_fp32 = np.float16(a).astype(np.float64)
    x_fp32 = np.float16(x).astype(np.float64)
    b_fp32 = np.float16(b).astype(np.float64)
    result = a_fp32 * x_fp32 + b_fp32
    return np.float16(result).astype(np.float64)


def matrix_multiply_with_bittrue_fma(X, W, Y):
    # Z = X @ W + Y as RedMulE does it: one FP16 FMA per k, in k order, starting from Y.
    # Vectorized over (m, n), bit-identical to calling bittrue_fma element by element.
    M, K = X.shape
    K2, N = W.shape
    assert K == K2, "Inner dimensions must match"

    X = np.asarray(X, dtype=np.float16)
    W = np.asarray(W, dtype=np.float16)
    acc = np.broadcast_to(np.asarray(Y, dtype=np.float16), (M, N)).astype(np.float64)
    for k in range(K):
        acc = bittrue_fma(X[:, k:k+1], W[k:k+1, :], acc)
    return acc.astype(np.float16)


def write_fp16_array(f, decl, A):
    # `uint16_t <decl> = { ... };` with the raw FP16 bits of A, one row of A per line
    A = np.asarray(A, dtype=np.float16)
    A = A.reshape(-1, A.shape[-1])
    f.write('uint16_t ' + decl + ' = {\n')
    rows, cols = A.shape
    for i in range(rows):
        for j in range(cols):
            f.write('0x' + hex(int(A[i, j].view('H')))[2:])
            if not (i == rows - 1 and j == cols - 1):
                f.write(', ')
        f.write('\n')
    f.write('};\n')
    f.write('\n')
