// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>
//
// L1 kernels called (synchronously, on the CV32 control core) by the
// mgc-generated test.c. All pointer arguments are L1 addresses.

#include <stdint.h>

/**
 * Copy `n` fp16 elements from `src` to `dst` (both in L1). A stand-in for a
 * software pre-processing step on the input block (e.g. a layout or precision
 * conversion) before RedMulE consumes it.
 */
void l1_copy_fp16(uint32_t dst, uint32_t src, uint32_t n)
{
    volatile uint32_t *d       = (volatile uint32_t *)dst;
    const volatile uint32_t *s = (const volatile uint32_t *)src;
    for (uint32_t k = 0; k < n / 2; k++) {
        d[k] = s[k];
    }
    if (n & 1) {
        ((volatile uint16_t *)dst)[n - 1] = ((const volatile uint16_t *)src)[n - 1];
    }
}
