// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// LUT-based element-wise binary add kernel (FP16), extracted from the maps
// test.

#include "add.h"

const uint16_t maps_generated_log_add_lut[8][8] = {
    {0x0000u, 0x398cu, 0x3c65u, 0x3d8cu, 0x3e70u, 0x3f2bu, 0x3fc9u, 0x4029u},
    {0x398cu, 0x3d8cu, 0x3f2bu, 0x4029u, 0x409bu, 0x40f8u, 0x4148u, 0x418cu},
    {0x3c65u, 0x3f2bu, 0x4065u, 0x40f8u, 0x416au, 0x41c8u, 0x4217u, 0x425cu},
    {0x3d8cu, 0x4029u, 0x40f8u, 0x418cu, 0x41feu, 0x425cu, 0x42aau, 0x42efu},
    {0x3e70u, 0x409bu, 0x416au, 0x41feu, 0x4270u, 0x42ceu, 0x431cu, 0x4361u},
    {0x3f2bu, 0x40f8u, 0x41c8u, 0x425cu, 0x42ceu, 0x432bu, 0x437au, 0x43beu},
    {0x3fc9u, 0x4148u, 0x4217u, 0x42aau, 0x431cu, 0x437au, 0x43c9u, 0x4407u},
    {0x4029u, 0x418cu, 0x425cu, 0x42efu, 0x4361u, 0x43beu, 0x4407u, 0x4429u},
};

void maps_add_f16(float16 *out, const float16 *lhs, const float16 *rhs, int32_t size)
{
    volatile uint16_t *dst        = (volatile uint16_t *)out;
    volatile const uint16_t *lhsp = (volatile const uint16_t *)lhs;
    volatile const uint16_t *rhsp = (volatile const uint16_t *)rhs;

    for (int32_t i = 0; i < size; ++i) {
        dst[i] = maps_generated_log_add_lut[maps_generated_log_index(lhsp[i])]
                                           [maps_generated_log_index(rhsp[i])];
    }
}
