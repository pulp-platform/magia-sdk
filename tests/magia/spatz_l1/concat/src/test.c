// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include "l1check.h"
#include "concat_fp16_spatz.h"

/* 37 iterations of 5 || 11 || 7 elements */
#define IT 37
#define NO (IT * 23)

uint16_t a[IT * 5], b[IT * 11], c[IT * 7], y_ref[NO];

int main(void)
{
    const uint32_t lens[3] = {5, 11, 7};
    const float16 *in[3]   = {(const float16 *)a, (const float16 *)b, (const float16 *)c};

    l1c_init();
    if (get_hartid() == 0) {
        l1c_fill(a, IT * 5, 1, 10, 17, 1);
        l1c_fill(b, IT * 11, 2, 10, 17, 1);
        l1c_fill(c, IT * 7, 3, 10, 17, 1);
    }
    l1c_sync();

    L1C_FULL(MAGIA_concat_fp16_spatz(in, lens, 3, (float16 *)y_ref, IT));
    l1c_sync();

    l1_alloc_init();
    uint16_t *a1 = l1c_alloc(IT * 5 * 2), *b1 = l1c_alloc(IT * 11 * 2), *c1 = l1c_alloc(IT * 7 * 2);
    uint16_t *y1 = l1c_alloc(NO * 2);
    const float16 *in1[3] = {(const float16 *)a1, (const float16 *)b1, (const float16 *)c1};
    l1c_copy_in(a1, a, IT * 5 * 2);
    l1c_copy_in(b1, b, IT * 11 * 2);
    l1c_copy_in(c1, c, IT * 7 * 2);
    L1C_L1(MAGIA_concat_fp16_spatz_l1(in1, lens, 3, (float16 *)y1, IT));

    uint32_t mx, err = l1c_compare(y1, y_ref, NO, 0, &mx);
    return l1c_report("concat", err, mx);
}
