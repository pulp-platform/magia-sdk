// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include "l1check.h"
#include "gemm_fp16_spatz.h"

/*
 * Y = 1.5 * A' @ B' + 0.5 * C, twice: M = 13, N = 37, K = 19 plain (the full API shards N), then
 * M = 37, N = 13, K = 19 with A and B transposed (it shards M).
 */
#define NA (37 * 19)
#define NB (19 * 37)
#define NY (13 * 37)

uint16_t a[NA], b[NB], c[NY], y_ref[2][NY];

int main(void)
{
    uint32_t a0[2] = {13, 19}, b0[2] = {19, 37}, y0[2] = {13, 37};
    uint32_t a1s[2] = {19, 37}, b1s[2] = {13, 19}, y1s[2] = {37, 13};

    l1c_init();
    if (get_hartid() == 0) {
        l1c_fill(a, NA, 1, 10, 15, 1);
        l1c_fill(b, NB, 2, 10, 15, 1);
        l1c_fill(c, NY, 3, 10, 15, 1);
    }
    l1c_sync();

    L1C_FULL(MAGIA_gemm_fp16_spatz((const float16 *)a, (const float16 *)b, (const float16 *)c, (float16)1.5f, (float16)0.5f, 0,
                          0, a0, b0, y0, y0, (float16 *)y_ref[0]));
    L1C_FULL(MAGIA_gemm_fp16_spatz((const float16 *)a, (const float16 *)b, (const float16 *)c, (float16)1.5f, (float16)0.5f, 1,
                          1, a1s, b1s, y1s, y1s, (float16 *)y_ref[1]));
    l1c_sync();

    l1_alloc_init();
    uint16_t *a1 = l1c_alloc(NA * 2), *b1 = l1c_alloc(NB * 2), *c1 = l1c_alloc(NY * 2), *y1 = l1c_alloc(2 * NY * 2);
    l1c_copy_in(a1, a, NA * 2);
    l1c_copy_in(b1, b, NB * 2);
    l1c_copy_in(c1, c, NY * 2);
    L1C_L1(MAGIA_gemm_fp16_spatz_l1((const float16 *)a1, (const float16 *)b1, (const float16 *)c1, (float16)1.5f,
                             (float16)0.5f, 0, 0, a0, b0, y0, y0, (float16 *)y1));
    L1C_L1(MAGIA_gemm_fp16_spatz_l1((const float16 *)a1, (const float16 *)b1, (const float16 *)c1, (float16)1.5f,
                             (float16)0.5f, 1, 1, a1s, b1s, y1s, y1s, (float16 *)(y1 + NY)));

    uint32_t mx, err = l1c_compare(y1, &y_ref[0][0], 2 * NY, 0, &mx);
    return l1c_report("gemm", err, mx);
}
