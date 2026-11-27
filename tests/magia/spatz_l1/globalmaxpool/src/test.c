// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include "l1check.h"
#include "globalmaxpool_fp16_spatz.h"

/* N = 2, C = 13 -> 26 planes of 7 x 9 */
#define NC 26
#define NX (NC * 7 * 9)

uint16_t x[NX], y_ref[NC];

int main(void)
{
    uint32_t shape[4] = {2, 13, 7, 9};

    l1c_init();
    if (get_hartid() == 0)
        l1c_fill(x, NX, 1, 10, 17, 1);
    l1c_sync();

    L1C_FULL(MAGIA_globalmaxpool_fp16_spatz((const float16 *)x, (float16 *)y_ref, shape));
    l1c_sync();

    l1_alloc_init();
    uint16_t *x1 = l1c_alloc(NX * 2), *y1 = l1c_alloc(NC * 2);
    l1c_copy_in(x1, x, NX * 2);
    L1C_L1(MAGIA_globalmaxpool_fp16_spatz_l1((const float16 *)x1, (float16 *)y1, shape));

    uint32_t mx, err = l1c_compare(y1, y_ref, NC, 0, &mx);
    return l1c_report("globalmaxpool", err, mx);
}
