// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include "l1check.h"
#include "batchnorm_fp16_spatz.h"

/* N = 2, C = 7 -> 14 planes of 5 x 7 */
#define C  7
#define NX (2 * C * 5 * 7)

uint16_t x[NX], g[C], b[C], m[C], v[C], y_ref[NX];

int main(void)
{
    uint32_t shape[4] = {2, C, 5, 7};

    l1c_init();
    if (get_hartid() == 0) {
        l1c_fill(x, NX, 1, 10, 17, 1);
        l1c_fill(g, C, 2, 12, 16, 1);
        l1c_fill(b, C, 3, 10, 15, 1);
        l1c_fill(m, C, 4, 10, 15, 1);
        l1c_fill(v, C, 5, 12, 17, 0);
    }
    l1c_sync();

    L1C_FULL(MAGIA_batchnorm_fp16_spatz((const float16 *)x, (const float16 *)g, (const float16 *)b, (const float16 *)m,
                               (const float16 *)v, (float16)1e-3f, (float16 *)y_ref, shape));
    l1c_sync();

    l1_alloc_init();
    uint16_t *x1 = l1c_alloc(NX * 2), *y1 = l1c_alloc(NX * 2);
    uint16_t *g1 = l1c_alloc(C * 2), *b1 = l1c_alloc(C * 2), *m1 = l1c_alloc(C * 2), *v1 = l1c_alloc(C * 2);
    l1c_copy_in(x1, x, NX * 2);
    l1c_copy_in(g1, g, C * 2);
    l1c_copy_in(b1, b, C * 2);
    l1c_copy_in(m1, m, C * 2);
    l1c_copy_in(v1, v, C * 2);
    L1C_L1(MAGIA_batchnorm_fp16_spatz_l1((const float16 *)x1, (const float16 *)g1, (const float16 *)b1, (const float16 *)m1,
                                  (const float16 *)v1, (float16)1e-3f, (float16 *)y1, shape));

    uint32_t mx, err = l1c_compare(y1, y_ref, NX, 0, &mx);
    return l1c_report("batchnorm", err, mx);
}
