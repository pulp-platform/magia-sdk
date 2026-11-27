// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include "l1check.h"
#include "matmul_fp16_spatz.h"

/* 5 batches of [7 x 9] @ [9 x 11]: once with A and B batched, once with B broadcast. */
#define BT 5
#define M  7
#define K  9
#define O  11
#define NA (BT * M * K)
#define NB (BT * K * O)
#define NY (BT * M * O)

uint16_t a[NA], b[NB], y_ref[NY], y_ref_bc[NY];

int main(void)
{
    l1c_init();
    if (get_hartid() == 0) {
        l1c_fill(a, NA, 1, 10, 15, 1);
        l1c_fill(b, NB, 2, 10, 15, 1);
    }
    l1c_sync();

    L1C_FULL(MAGIA_matmul_fp16_spatz((const float16 *)a, (const float16 *)b, (float16 *)y_ref, M, K, O, BT, 1, 1));
    L1C_FULL(MAGIA_matmul_fp16_spatz((const float16 *)a, (const float16 *)b, (float16 *)y_ref_bc, M, K, O, BT, 1, 0));
    l1c_sync();

    l1_alloc_init();
    uint16_t *a1 = l1c_alloc(NA * 2), *b1 = l1c_alloc(NB * 2), *y1 = l1c_alloc(NY * 2), *y2 = l1c_alloc(NY * 2);
    l1c_copy_in(a1, a, NA * 2);
    l1c_copy_in(b1, b, NB * 2);
    L1C_L1(MAGIA_matmul_fp16_spatz_l1((const float16 *)a1, (const float16 *)b1, (float16 *)y1, M, K, O, BT, 1, 1));
    L1C_L1(MAGIA_matmul_fp16_spatz_l1((const float16 *)a1, (const float16 *)b1, (float16 *)y2, M, K, O, BT, 1, 0));

    uint32_t mx, mx2, err = l1c_compare(y1, y_ref, NY, 0, &mx);
    err += l1c_compare(y2, y_ref_bc, NY, 0, &mx2);
    return l1c_report("matmul", err, mx > mx2 ? mx : mx2);
}
