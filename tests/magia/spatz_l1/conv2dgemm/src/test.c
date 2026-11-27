// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include "l1check.h"
#include "conv2dgemm_fp16_spatz.h"

/*
 * X [1, 3, 11, 9], W [13, 3, 3, 3], bias, stride 1, pad 1 -> Y [1, 13, 11, 9]. The full API
 * shards the output rows on 4 tiles and the output channels on 16.
 */
#define NX (3 * 11 * 9)
#define NW (13 * 27)
#define NY (13 * 11 * 9)
#define NS (27 * 99 + NY)

uint16_t x[NX], w[NW], b[13], y_ref[NY];

int main(void)
{
    uint32_t in_shape[4] = {1, 3, 11, 9}, out_shape[4] = {1, 13, 11, 9};

    l1c_init();
    if (get_hartid() == 0) {
        l1c_fill(x, NX, 1, 10, 15, 1);
        l1c_fill(w, NW, 2, 10, 14, 1);
        l1c_fill(b, 13, 3, 10, 15, 1);
    }
    l1c_sync();

    L1C_FULL(MAGIA_conv2dgemm_fp16_spatz((const float16 *)x, (const float16 *)w, (const float16 *)b, (float16 *)y_ref,
                                in_shape, out_shape, 3, 3, 1, 1, 1, 1, 1, 1));
    l1c_sync();

    l1_alloc_init();
    uint16_t *x1 = l1c_alloc(NX * 2), *w1 = l1c_alloc(NW * 2), *b1 = l1c_alloc(13 * 2), *y1 = l1c_alloc(NY * 2);
    uint16_t *s1 = l1c_alloc(NS * 2);
    l1c_copy_in(x1, x, NX * 2);
    l1c_copy_in(w1, w, NW * 2);
    l1c_copy_in(b1, b, 13 * 2);
    L1C_L1(MAGIA_conv2dgemm_fp16_spatz_l1((const float16 *)x1, (const float16 *)w1, (const float16 *)b1, (float16 *)y1,
                                   in_shape, out_shape, 3, 3, 1, 1, 1, 1, 1, 1, (float16 *)s1));

    uint32_t mx, err = l1c_compare(y1, y_ref, NY, 0, &mx);
    return l1c_report("conv2dgemm", err, mx);
}
