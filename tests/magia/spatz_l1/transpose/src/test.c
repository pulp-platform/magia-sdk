// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include "l1check.h"
#include "transpose_fp16_spatz.h"

/* [13, 3, 5, 6] -> perm (0, 3, 1, 2) -> [13, 6, 3, 5]; the kernel requires perm[0] == 0. */
#define N (13 * 3 * 5 * 6)

uint16_t x[N], y_ref[N];
uint32_t perm[4] = {0, 3, 1, 2}, in_shape[4] = {13, 3, 5, 6}, out_shape[4] = {13, 6, 3, 5};

int main(void)
{
    l1c_init();
    if (get_hartid() == 0)
        l1c_fill(x, N, 1, 10, 17, 1);
    l1c_sync();

    L1C_FULL(MAGIA_transpose_fp16_spatz((const float16 *)x, (float16 *)y_ref, perm, in_shape, out_shape, 4, 13));
    l1c_sync();

    l1_alloc_init();
    uint16_t *x1 = l1c_alloc(N * 2), *y1 = l1c_alloc(N * 2);
    l1c_copy_in(x1, x, N * 2);
    L1C_L1(MAGIA_transpose_fp16_spatz_l1((const float16 *)x1, (float16 *)y1, perm, in_shape, out_shape, 4, 13));

    uint32_t mx, err = l1c_compare(y1, y_ref, N, 0, &mx);
    return l1c_report("transpose", err, mx);
}
