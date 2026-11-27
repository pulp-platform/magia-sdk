// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include "l1check.h"
#include "groupnorm_fp16_spatz.h"

/* X [3, 6, 5, 7], 3 groups of 2 channels -> 9 groups */
#define C  6
#define NX (3 * C * 5 * 7)

uint16_t x[NX], g[C], b[C], y_ref[NX];

int main(void)
{
    uint32_t shape[4] = {3, C, 5, 7};

    l1c_init();
    if (get_hartid() == 0) {
        l1c_fill(x, NX, 1, 10, 17, 1);
        l1c_fill(g, C, 2, 12, 16, 1);
        l1c_fill(b, C, 3, 10, 15, 1);
    }
    l1c_sync();

    L1C_FULL(MAGIA_groupnorm_fp16_spatz((const float16 *)x, (const float16 *)y_ref, (const float16 *)g, (const float16 *)b,
                               shape, 3, (float16)1e-3f));
    l1c_sync();

    l1_alloc_init();
    uint16_t *x1 = l1c_alloc(NX * 2), *y1 = l1c_alloc(NX * 2), *g1 = l1c_alloc(C * 2), *b1 = l1c_alloc(C * 2);
    l1c_copy_in(x1, x, NX * 2);
    l1c_copy_in(g1, g, C * 2);
    l1c_copy_in(b1, b, C * 2);
    L1C_L1(MAGIA_groupnorm_fp16_spatz_l1((const float16 *)x1, (float16 *)y1, (const float16 *)g1, (const float16 *)b1, shape,
                                  3, (float16)1e-3f));

    uint32_t mx, err = l1c_compare(y1, y_ref, NX, 0, &mx);
    return l1c_report("groupnorm", err, mx);
}
