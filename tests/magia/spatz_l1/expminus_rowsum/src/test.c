// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include "l1check.h"
#include "expminus_rowsum_fp16_spatz.h"

/* cols a multiple of 4 (fast path); X - M and A - M in (-10, 0) */
#define ROWS 37
#define COLS 40
#define SCALE 0.5f

/* A is in place: a_ref is the full API's copy, a the L1 API's input */
uint16_t x[ROWS * COLS], m[ROWS], a[ROWS], a_ref[ROWS], y_ref[ROWS * COLS], s_ref[ROWS];

int main(void)
{
    l1c_init();
    if (get_hartid() == 0) {
        l1c_fill(x, ROWS * COLS, 1, 10, 15, 1);
        l1c_fill(m, ROWS, 2, 16, 17, 0);
        l1c_fill(a, ROWS, 3, 10, 15, 1);
        l1c_fill(a_ref, ROWS, 3, 10, 15, 1);
    }
    l1c_sync();

    L1C_FULL(MAGIA_expminus_rowsum_fp16_spatz((const float16 *)x, (const float16 *)m, (float16 *)y_ref,
                                     (float16 *)s_ref, (float16 *)a_ref, SCALE, ROWS, COLS));
    l1c_sync();

    l1_alloc_init();
    uint16_t *x1 = l1c_alloc(ROWS * COLS * 2), *m1 = l1c_alloc(ROWS * 2), *a1 = l1c_alloc(ROWS * 2);
    uint16_t *y1 = l1c_alloc(ROWS * COLS * 2), *s1 = l1c_alloc(ROWS * 2);
    l1c_copy_in(x1, x, ROWS * COLS * 2);
    l1c_copy_in(m1, m, ROWS * 2);
    l1c_copy_in(a1, a, ROWS * 2);
    L1C_L1(MAGIA_expminus_rowsum_fp16_spatz_l1((const float16 *)x1, (const float16 *)m1, (float16 *)y1,
                                        (float16 *)s1, (float16 *)a1, SCALE, ROWS, COLS));

    uint32_t mx_y, mx_s, mx_a, err = l1c_compare(y1, y_ref, ROWS * COLS, 0, &mx_y);
    err += l1c_compare(s1, s_ref, ROWS, 0, &mx_s);
    err += l1c_compare(a1, a_ref, ROWS, 0, &mx_a);
    uint32_t mx = mx_y > mx_s ? mx_y : mx_s;
    return l1c_report("expminus_rowsum", err, mx > mx_a ? mx : mx_a);
}
