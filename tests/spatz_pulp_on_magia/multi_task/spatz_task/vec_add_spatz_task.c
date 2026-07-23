/*
 * Copyright (C) 2025 ETH Zurich and University of Bologna
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "tile.h"
#include "multi_task_params.h"

int vec_add_spatz_task(void)
{
    uintptr_t params_addr               = mmio32(SPATZ_DATA);
    volatile spatz_add_params_t *params = (volatile spatz_add_params_t *)params_addr;
    _Float16 *A                         = (_Float16 *)params->A;
    _Float16 *B                         = (_Float16 *)params->B;
    _Float16 *C                         = (_Float16 *)params->C;
    size_t avl                          = params->len;
    size_t vl;

    for (; avl > 0; avl -= vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile("vle16.v v0, (%0)" ::"r"(A));
        asm volatile("vle16.v v8, (%0)" ::"r"(B));
        asm volatile("vfadd.vv v16, v0, v8");
        asm volatile("vse16.v v16, (%0)" ::"r"(C));
        A += vl;
        B += vl;
        C += vl;
    }

    return 0;
}
