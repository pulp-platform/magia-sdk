// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include "l1check.h"
#include "rowmax_fp16_spatz.h"

/* cols even: the vector path (odd cols would take the strided fallback) */
#define ROWS 37
#define COLS 40

uint16_t x[ROWS * COLS], p[ROWS], m_ref[ROWS];

int main(void)
{
    l1c_init();
    if (get_hartid() == 0) {
        l1c_fill(x, ROWS * COLS, 1, 10, 17, 1);
        l1c_fill(p, ROWS, 2, 10, 17, 1);
    }
    l1c_sync();

    L1C_FULL(MAGIA_rowmax_fp16_spatz((const float16 *)x, (const float16 *)p, (float16 *)m_ref, ROWS, COLS));
    l1c_sync();

    l1_alloc_init();
    uint16_t *x1 = l1c_alloc(ROWS * COLS * 2), *p1 = l1c_alloc(ROWS * 2), *m1 = l1c_alloc(ROWS * 2);
    l1c_copy_in(x1, x, ROWS * COLS * 2);
    l1c_copy_in(p1, p, ROWS * 2);
    L1C_L1(MAGIA_rowmax_fp16_spatz_l1((const float16 *)x1, (const float16 *)p1, (float16 *)m1, ROWS, COLS));

    uint32_t mx, err = l1c_compare(m1, m_ref, ROWS, 0, &mx);
    return l1c_report("rowmax", err, mx);
}
