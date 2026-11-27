// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include "l1check.h"
#include "rmsnorm_fp16_spatz.h"

#define ROWS 37
#define W    27
#define N    (ROWS * W)

uint16_t x[N], g[W], y_ref[N];

int main(void)
{
    l1c_init();
    if (get_hartid() == 0) {
        l1c_fill(x, N, 1, 10, 17, 1);
        l1c_fill(g, W, 2, 13, 16, 1);
    }
    l1c_sync();

    L1C_FULL(MAGIA_rmsnorm_fp16_spatz((const float16 *)x, (const float16 *)g, (float16)1e-3f, ROWS, W, (float16 *)y_ref));
    l1c_sync();

    l1_alloc_init();
    uint16_t *x1 = l1c_alloc(N * 2), *g1 = l1c_alloc(W * 2), *y1 = l1c_alloc(N * 2);
    float16 *e1 = l1c_alloc(2);
    *e1 = (float16)1e-3f;
    l1c_copy_in(x1, x, N * 2);
    l1c_copy_in(g1, g, W * 2);
    L1C_L1(MAGIA_rmsnorm_fp16_spatz_l1((const float16 *)x1, (const float16 *)g1, e1, ROWS, W, (float16 *)y1));

    uint32_t mx, err = l1c_compare(y1, y_ref, N, 0, &mx);
    return l1c_report("rmsnorm", err, mx);
}
