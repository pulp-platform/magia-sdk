// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include "l1check.h"
#include "convtranspose_fp16_spatz.h"

/* X [1, 4, 5, 6], W [4, 13, 3, 3], stride 2, pad 1 -> Y [1, 13, 9, 11] */
#define NX (4 * 5 * 6)
#define NW (4 * 13 * 9)
#define NY (13 * 9 * 11)

uint16_t x[NX], w[NW], y_ref[NY];

int main(void)
{
    uint32_t in_shape[4] = {1, 4, 5, 6}, out_shape[4] = {1, 13, 9, 11};

    l1c_init();
    if (get_hartid() == 0) {
        l1c_fill(x, NX, 1, 10, 15, 1);
        l1c_fill(w, NW, 2, 10, 14, 1);
    }
    l1c_sync();

    L1C_FULL(MAGIA_convtranspose_fp16_spatz((const float16 *)x, (const float16 *)w, (float16 *)y_ref, in_shape, out_shape, 3,
                                   3, 2, 2, 1, 1, 1));
    l1c_sync();

    l1_alloc_init();
    uint16_t *x1 = l1c_alloc(NX * 2), *w1 = l1c_alloc(NW * 2), *y1 = l1c_alloc(NY * 2);
    l1c_copy_in(x1, x, NX * 2);
    l1c_copy_in(w1, w, NW * 2);
    L1C_L1(MAGIA_convtranspose_fp16_spatz_l1((const float16 *)x1, (const float16 *)w1, (float16 *)y1, in_shape, out_shape,
                                      3, 3, 2, 2, 1, 1, 1));

    uint32_t mx, err = l1c_compare(y1, y_ref, NY, 0, &mx);
    return l1c_report("convtranspose", err, mx);
}
