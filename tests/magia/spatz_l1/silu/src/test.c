// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include "l1check.h"
#include "silu_fp16_spatz.h"

#define N 1001

uint16_t x[N], y_ref[N];

int main(void)
{
    l1c_init();
    if (get_hartid() == 0)
        l1c_fill(x, N, 1, 10, 17, 1);
    l1c_sync();

    L1C_FULL(MAGIA_silu_fp16_spatz((const float16 *)x, (float16 *)y_ref, N));
    l1c_sync();

    l1_alloc_init();
    uint16_t *x1 = l1c_alloc(N * 2), *y1 = l1c_alloc(N * 2);
    l1c_copy_in(x1, x, N * 2);
    L1C_L1(MAGIA_silu_fp16_spatz_l1((const float16 *)x1, (float16 *)y1, N));

    uint32_t mx, err = l1c_compare(y1, y_ref, N, 0, &mx);
    return l1c_report("silu", err, mx);
}
