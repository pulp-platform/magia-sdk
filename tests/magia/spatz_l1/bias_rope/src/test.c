// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include "l1check.h"
#include "bias_rope_fp16_spatz.h"

/* cols a multiple of GROUP; the block starts at COL0 of its head and straddles two heads */
#define ROWS     37
#define COLS     48
#define COL0     16
#define HEAD_DIM 32
#define GROUP    8
#define HALF     (HEAD_DIM / 2)

/* X is in place: x_ref is the full API's copy, x the L1 API's input */
uint16_t x[ROWS * COLS], x_ref[ROWS * COLS], b[COLS], c[ROWS * HALF], s[ROWS * HALF];

int main(void)
{
    l1c_init();
    if (get_hartid() == 0) {
        l1c_fill(x, ROWS * COLS, 1, 10, 17, 1);
        l1c_fill(x_ref, ROWS * COLS, 1, 10, 17, 1);
        l1c_fill(b, COLS, 2, 10, 15, 1);
        l1c_fill(c, ROWS * HALF, 3, 10, 14, 1);
        l1c_fill(s, ROWS * HALF, 4, 10, 14, 1);
    }
    l1c_sync();

    L1C_FULL(MAGIA_bias_rope_fp16_spatz((float16 *)x_ref, (const float16 *)b, (const float16 *)c,
                               (const float16 *)s, ROWS, COLS, COL0, HEAD_DIM, GROUP));
    l1c_sync();

    l1_alloc_init();
    uint16_t *x1 = l1c_alloc(ROWS * COLS * 2), *b1 = l1c_alloc(COLS * 2);
    uint16_t *c1 = l1c_alloc(ROWS * HALF * 2), *s1 = l1c_alloc(ROWS * HALF * 2);
    l1c_copy_in(x1, x, ROWS * COLS * 2);
    l1c_copy_in(b1, b, COLS * 2);
    l1c_copy_in(c1, c, ROWS * HALF * 2);
    l1c_copy_in(s1, s, ROWS * HALF * 2);
    L1C_L1(MAGIA_bias_rope_fp16_spatz_l1((float16 *)x1, (const float16 *)b1, (const float16 *)c1,
                                  (const float16 *)s1, ROWS, COLS, COL0, HEAD_DIM, GROUP));

    uint32_t mx, err = l1c_compare(x1, x_ref, ROWS * COLS, 0, &mx);
    return l1c_report("bias_rope", err, mx);
}
