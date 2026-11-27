// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include "l1check.h"
#include "softmax_rows_fp16_spatz.h"

/* cols even */
#define ROWS 37
#define COLS 40
#define SCALE 0.5f

/* X is in place: x_ref is the full API's copy, x the L1 API's input */
uint16_t x[ROWS * COLS], x_ref[ROWS * COLS];

int main(void)
{
    l1c_init();
    if (get_hartid() == 0) {
        l1c_fill(x, ROWS * COLS, 1, 10, 17, 1);
        l1c_fill(x_ref, ROWS * COLS, 1, 10, 17, 1);
    }
    l1c_sync();

    L1C_FULL(MAGIA_softmax_rows_fp16_spatz((float16 *)x_ref, SCALE, ROWS, COLS));
    l1c_sync();

    l1_alloc_init();
    uint16_t *x1 = l1c_alloc(ROWS * COLS * 2);
    l1c_copy_in(x1, x, ROWS * COLS * 2);
    L1C_L1(MAGIA_softmax_rows_fp16_spatz_l1((float16 *)x1, SCALE, ROWS, COLS));

    uint32_t mx, err = l1c_compare(x1, x_ref, ROWS * COLS, 0, &mx);
    return l1c_report("softmax_rows", err, mx);
}
