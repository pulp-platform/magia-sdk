// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include "l1check.h"
#include "rowdiv_fp16_spatz.h"

#define ROWS 37
#define COLS 27
#define N    (ROWS * COLS)

uint16_t x[N], s[ROWS], y_ref[N];

int main(void)
{
    l1c_init();
    if (get_hartid() == 0) {
        l1c_fill(x, N, 1, 10, 17, 1);
        l1c_fill(s, ROWS, 2, 13, 16, 1);
    }
    l1c_sync();

    L1C_FULL(MAGIA_rowdiv_fp16_spatz((const float16 *)x, (const float16 *)s, (float16 *)y_ref, ROWS, COLS));
    l1c_sync();

    l1_alloc_init();
    uint16_t *x1 = l1c_alloc(N * 2), *s1 = l1c_alloc(ROWS * 2), *y1 = l1c_alloc(N * 2);
    l1c_copy_in(x1, x, N * 2);
    l1c_copy_in(s1, s, ROWS * 2);
    L1C_L1(MAGIA_rowdiv_fp16_spatz_l1((const float16 *)x1, (const float16 *)s1, (float16 *)y1, ROWS, COLS));

    uint32_t mx, err = l1c_compare(y1, y_ref, N, 0, &mx);
    return l1c_report("rowdiv", err, mx);
}
