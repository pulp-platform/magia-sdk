// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// RedMulE-backed kernels extracted from the maps test.

#include "redmule_kernels.h"

void maps_matmul_redmule(redmule_controller_t *redmule,
                         eu_controller_t *eu,
                         uint32_t x,
                         uint32_t w,
                         uint32_t y,
                         uint16_t m,
                         uint16_t n,
                         uint16_t k,
                         eu_wait_mode_t wait_mode)
{
    redmule_gemm(redmule, x, w, y, m, n, k);
    eu_redmule_wait(eu, wait_mode);
}
