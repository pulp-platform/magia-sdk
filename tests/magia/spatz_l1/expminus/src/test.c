// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include "l1check.h"
#include "expminus_fp16_spatz.h"

#define R  37
#define CO 27
#define NX (R * CO)

uint16_t x[NX], m[R], y_ref[NX];

int main(void)
{
    l1c_init();
    if (get_hartid() == 0) {
        l1c_fill(x, NX, 1, 8, 16, 1); /* |x| < 8 */
        l1c_fill(m, R, 2, 17, 17, 0); /* m in [8, 16) >= x */
    }
    l1c_sync();

    L1C_FULL(MAGIA_expminus_fp16_spatz((const float16 *)x, (const float16 *)m, (float16 *)y_ref, R, CO));
    l1c_sync();

    l1_alloc_init();
    uint16_t *x1 = l1c_alloc(NX * 2), *m1 = l1c_alloc(R * 2), *y1 = l1c_alloc(NX * 2);
    l1c_copy_in(x1, x, NX * 2);
    l1c_copy_in(m1, m, R * 2);
    L1C_L1(MAGIA_expminus_fp16_spatz_l1((const float16 *)x1, (const float16 *)m1, (float16 *)y1, R, CO));

    uint32_t mx, err = l1c_compare(y1, y_ref, NX, 0, &mx);
    return l1c_report("expminus", err, mx);
}
