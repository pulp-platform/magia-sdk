// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include "l1check.h"
#include "colsum_rescale_fp16_spatz.h"

/* cols even (37 column pairs over the tiles); row 0 of G is unused */
#define ROWS 13
#define COLS 74

/* S and P are in place: s_ref / p_ref are the full API's copies, s / p the L1 API's inputs */
uint16_t s[COLS], g[ROWS * COLS], p[COLS], a[COLS], s_ref[COLS], p_ref[COLS];

int main(void)
{
    l1c_init();
    if (get_hartid() == 0) {
        l1c_fill(s, COLS, 1, 10, 16, 0);
        l1c_fill(s_ref, COLS, 1, 10, 16, 0);
        l1c_fill(g, ROWS * COLS, 2, 10, 16, 0);
        l1c_fill(p, COLS, 3, 10, 16, 0);
        l1c_fill(p_ref, COLS, 3, 10, 16, 0);
        l1c_fill(a, COLS, 4, 10, 14, 0);
    }
    l1c_sync();

    L1C_FULL(MAGIA_colsum_rescale_fp16_spatz(
        (float16 *)s_ref, (const float16 *)g, (float16 *)p_ref, (const float16 *)a, ROWS, COLS));
    l1c_sync();

    l1_alloc_init();
    uint16_t *s1 = l1c_alloc(COLS * 2), *g1 = l1c_alloc(ROWS * COLS * 2);
    uint16_t *p1 = l1c_alloc(COLS * 2), *a1 = l1c_alloc(COLS * 2);
    l1c_copy_in(s1, s, COLS * 2);
    l1c_copy_in(g1, g, ROWS * COLS * 2);
    l1c_copy_in(p1, p, COLS * 2);
    l1c_copy_in(a1, a, COLS * 2);
    L1C_L1(MAGIA_colsum_rescale_fp16_spatz_l1(
        (float16 *)s1, (const float16 *)g1, (float16 *)p1, (const float16 *)a1, ROWS, COLS));

    uint32_t mx, mx_p, err = l1c_compare(s1, s_ref, COLS, 0, &mx);
    err += l1c_compare(p1, p_ref, COLS, 0, &mx_p);
    return l1c_report("colsum_rescale", err, mx > mx_p ? mx : mx_p);
}
