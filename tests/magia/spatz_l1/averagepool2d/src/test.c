// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include "l1check.h"
#include "averagepool2d_fp16_spatz.h"

/* 2 x 7 = 14 planes of 11 x 9, 3x3 window, stride 2, pad 1 -> 6 x 5 */
#define NC 14
#define NX (NC * 11 * 9)
#define NY (NC * 6 * 5)

uint16_t x[NX], y_ref[NY];

int main(void)
{
    uint32_t in_shape[4] = {2, 7, 11, 9}, out_shape[4] = {2, 7, 6, 5};

    l1c_init();
    if (get_hartid() == 0)
        l1c_fill(x, NX, 1, 10, 17, 1);
    l1c_sync();

    L1C_FULL(MAGIA_averagepool2d_fp16_spatz((const float16 *)x, (float16 *)y_ref, in_shape, out_shape, 3, 3, 2, 2, 1, 1));
    l1c_sync();

    l1_alloc_init();
    uint16_t *x1 = l1c_alloc(NX * 2), *y1 = l1c_alloc(NY * 2);
    l1c_copy_in(x1, x, NX * 2);
    L1C_L1(MAGIA_averagepool2d_fp16_spatz_l1((const float16 *)x1, (float16 *)y1, in_shape, out_shape, 3, 3, 2, 2, 1, 1));

    uint32_t mx, err = l1c_compare(y1, y_ref, NY, 0, &mx);
    return l1c_report("averagepool2d", err, mx);
}
