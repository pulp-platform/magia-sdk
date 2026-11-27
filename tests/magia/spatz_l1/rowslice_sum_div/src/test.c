// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include "l1check.h"
#include "rowslice_sum_div_fp16_spatz.h"

/* cols even (4-byte aligned rows) */
#define N    3
#define OWN  1
#define ROWS 37
#define COLS 20

uint16_t o[ROWS * COLS], g[N * ROWS * COLS], s[ROWS], y_ref[ROWS * COLS];

int main(void)
{
    l1c_init();
    if (get_hartid() == 0) {
        l1c_fill(o, ROWS * COLS, 1, 10, 17, 1);
        l1c_fill(g, N * ROWS * COLS, 2, 10, 17, 1);
        l1c_fill(s, ROWS, 3, 14, 17, 0);
    }
    l1c_sync();

    L1C_FULL(MAGIA_rowslice_sum_div_fp16_spatz((float16 *)y_ref, (const float16 *)o, (const float16 *)g,
                                      (const float16 *)s, N, OWN, ROWS, COLS));
    l1c_sync();

    l1_alloc_init();
    uint16_t *o1 = l1c_alloc(ROWS * COLS * 2), *g1 = l1c_alloc(N * ROWS * COLS * 2);
    uint16_t *s1 = l1c_alloc(ROWS * 2), *y1 = l1c_alloc(ROWS * COLS * 2);
    l1c_copy_in(o1, o, ROWS * COLS * 2);
    l1c_copy_in(g1, g, N * ROWS * COLS * 2);
    l1c_copy_in(s1, s, ROWS * 2);
    L1C_L1(MAGIA_rowslice_sum_div_fp16_spatz_l1((float16 *)y1, (const float16 *)o1, (const float16 *)g1,
                                         (const float16 *)s1, N, OWN, ROWS, COLS));

    uint32_t mx, err = l1c_compare(y1, y_ref, ROWS * COLS, 0, &mx);
    return l1c_report("rowslice_sum_div", err, mx);
}
