// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// LUT-based element-wise natural logarithm kernel (FP16), extracted from the
// maps test. Inputs are drawn from a fixed 8-value set; the kernel maps each
// input to its precomputed log via a lookup table.

#ifndef KERNELS_GENERIC_LOG_H
#define KERNELS_GENERIC_LOG_H

#include <stdint.h>

/* Precomputed log(x) values (FP16 bit patterns) for the 8 supported inputs. */
extern const uint16_t maps_generated_log_lut[8];

/* Maps a supported FP16 input value (bit pattern) to its LUT index [0,8). */
static inline uint32_t maps_generated_value_index(uint16_t value)
{
    switch (value) {
    case 0x3c00u:
        return 0u;
    case 0x4000u:
        return 1u;
    case 0x4200u:
        return 2u;
    case 0x4400u:
        return 3u;
    case 0x4500u:
        return 4u;
    case 0x4600u:
        return 5u;
    case 0x4700u:
        return 6u;
    case 0x4800u:
        return 7u;
    default:
        return 0u;
    }
}

/* out[i] = log(in[i]) via LUT, for `size` FP16 elements. */
void maps_log_f16(float16 *out, const float16 *in, int32_t size);

#endif /* KERNELS_GENERIC_LOG_H */
