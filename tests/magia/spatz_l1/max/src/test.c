// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include "l1check.h"
#include "max_fp16_spatz.h"

#define N 1001

uint16_t a[N], b[N], c_ref[N];

int main(void)
{
    l1c_init();
    if (get_hartid() == 0) {
        l1c_fill(a, N, 1, 10, 17, 1);
        l1c_fill(b, N, 2, 10, 17, 1);
    }
    l1c_sync();

    L1C_FULL(MAGIA_max_fp16_spatz((const float16 *)a, (const float16 *)b, (float16 *)c_ref, N));
    l1c_sync();

    l1_alloc_init();
    uint16_t *a1 = l1c_alloc(N * 2), *b1 = l1c_alloc(N * 2), *c1 = l1c_alloc(N * 2);
    l1c_copy_in(a1, a, N * 2);
    l1c_copy_in(b1, b, N * 2);
    L1C_L1(MAGIA_max_fp16_spatz_l1((const float16 *)a1, (const float16 *)b1, (float16 *)c1, N));

    uint32_t mx, err = l1c_compare(c1, c_ref, N, 0, &mx);
    return l1c_report("max", err, mx);
}
