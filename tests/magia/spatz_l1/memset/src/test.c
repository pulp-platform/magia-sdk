// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include "l1check.h"
#include "memset_fp16_spatz.h"

/* memset has no full API: the reference is N zeros followed by one untouched guard element. */
#define N 1001

uint16_t x[N + 1], y_ref[N + 1];

int main(void)
{
    l1c_init();
    if (get_hartid() == 0) {
        l1c_fill(x, N + 1, 1, 10, 17, 1);
        y_ref[N] = x[N];
    }
    l1c_sync();

    l1_alloc_init();
    uint16_t *y1 = l1c_alloc((N + 1) * 2);
    l1c_copy_in(y1, x, (N + 1) * 2);
    L1C_L1(MAGIA_memset_zero_fp16_spatz_l1((float16 *)y1, N));

    uint32_t mx, err = l1c_compare(y1, y_ref, N + 1, 0, &mx);
    return l1c_report("memset", err, mx);
}
