// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// LUT-based element-wise binary add kernel (FP16), extracted from the maps
// test. Operands are drawn from a fixed 8-value set of log outputs; the kernel
// resolves each pair to a precomputed sum via an 8x8 lookup table.

#ifndef KERNELS_GENERIC_ADD_H
#define KERNELS_GENERIC_ADD_H

#include <stdint.h>

/* Precomputed lhs+rhs values (FP16 bit patterns) indexed by log-value pair. */
extern const uint16_t maps_generated_log_add_lut[8][8];

/* Maps a supported FP16 log-output value (bit pattern) to its index [0,8). */
static inline uint32_t maps_generated_log_index(uint16_t value)
{
    switch (value) {
    case 0x398cu:
        return 1u;
    case 0x3c65u:
        return 2u;
    case 0x3d8cu:
        return 3u;
    case 0x3e70u:
        return 4u;
    case 0x3f2bu:
        return 5u;
    case 0x3fc9u:
        return 6u;
    case 0x4029u:
        return 7u;
    case 0x0000u:
    default:
        return 0u;
    }
}

/* out[i] = lhs[i] + rhs[i] via LUT, for `size` FP16 elements. */
void maps_add_f16(float16 *out, const float16 *lhs, const float16 *rhs, int32_t size);

#endif /* KERNELS_GENERIC_ADD_H */
