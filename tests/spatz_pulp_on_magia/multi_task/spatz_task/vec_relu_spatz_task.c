/*
 * Copyright (C) 2025 ETH Zurich and University of Bologna
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "tile.h"
#include "multi_task_params.h"

int vec_relu_spatz_task(void)
{
    register _Float16 ZERO asm ("fs0") = 0.0f;
    uintptr_t params_addr = mmio32(SPATZ_DATA);
    volatile spatz_relu_params_t *params = (volatile spatz_relu_params_t *)params_addr;
    _Float16 *X = (_Float16 *)params->X;
    _Float16 *Y = (_Float16 *)params->Y;
    size_t avl = params->len;
    size_t vl;

    for (; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile ("vle16.v v0, (%0)" :: "r"(X));
        asm volatile ("vfmax.vf v8, v0, %0" :: "f"(ZERO));
        asm volatile ("vse16.v v8, (%0)" :: "r"(Y));
        X += vl;
        Y += vl;
    }

    return 0;
}
