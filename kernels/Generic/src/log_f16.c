// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// LUT-based element-wise natural logarithm kernel (FP16), extracted from the
// maps test.

#include "log.h"

const uint16_t maps_generated_log_lut[8] = {
    0x0000u,
    0x398cu,
    0x3c65u,
    0x3d8cu,
    0x3e70u,
    0x3f2bu,
    0x3fc9u,
    0x4029u,
};

void maps_log_f16(float16 *out, const float16 *in, int32_t size)
{
    volatile uint16_t *dst       = (volatile uint16_t *)out;
    volatile const uint16_t *src = (volatile const uint16_t *)in;

    for (int32_t i = 0; i < size; ++i) {
        dst[i] = maps_generated_log_lut[maps_generated_value_index(src[i])];
    }
}
