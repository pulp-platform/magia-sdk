// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include "l1check.h"
#include "colmax_fp16_spatz.h"

/* cols even (37 column pairs over the tiles) */
#define ROWS 13
#define COLS 74

uint16_t x[ROWS * COLS], m_ref[COLS];

int main(void)
{
    l1c_init();
    if (get_hartid() == 0)
        l1c_fill(x, ROWS * COLS, 1, 10, 17, 1);
    l1c_sync();

    L1C_FULL(MAGIA_colmax_fp16_spatz((const float16 *)x, (float16 *)m_ref, ROWS, COLS));
    l1c_sync();

    l1_alloc_init();
    uint16_t *x1 = l1c_alloc(ROWS * COLS * 2), *m1 = l1c_alloc(COLS * 2);
    l1c_copy_in(x1, x, ROWS * COLS * 2);
    L1C_L1(MAGIA_colmax_fp16_spatz_l1((const float16 *)x1, (float16 *)m1, ROWS, COLS));

    uint32_t mx, err = l1c_compare(m1, m_ref, COLS, 0, &mx);
    return l1c_report("colmax", err, mx);
}
