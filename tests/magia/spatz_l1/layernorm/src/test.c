// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include "l1check.h"
#include "layernorm_fp16_spatz.h"

/* [3, 13, 27]: 39 rows of 27 */
#define W 27
#define N (3 * 13 * W)

uint16_t x[N], g[W], b[W], y_ref[N];
uint32_t shape[3] = {3, 13, W};

int main(void)
{
    l1c_init();
    if (get_hartid() == 0) {
        l1c_fill(x, N, 1, 10, 17, 1);
        l1c_fill(g, W, 2, 13, 16, 1);
        l1c_fill(b, W, 3, 10, 15, 1);
    }
    l1c_sync();

    L1C_FULL(MAGIA_layernorm_fp16_spatz((const float16 *)x, (const float16 *)g, (const float16 *)b, (float16)1e-3f, shape, 3,
                               (float16 *)y_ref));
    l1c_sync();

    l1_alloc_init();
    uint16_t *x1 = l1c_alloc(N * 2), *g1 = l1c_alloc(W * 2), *b1 = l1c_alloc(W * 2), *y1 = l1c_alloc(N * 2);
    l1c_copy_in(x1, x, N * 2);
    l1c_copy_in(g1, g, W * 2);
    l1c_copy_in(b1, b, W * 2);
    L1C_L1(MAGIA_layernorm_fp16_spatz_l1((const float16 *)x1, (const float16 *)g1, (const float16 *)b1, (float16)1e-3f, shape, 3,
                                  (float16 *)y1));

    uint32_t mx, err = l1c_compare(y1, y_ref, N, 0, &mx);
    return l1c_report("layernorm", err, mx);
}
