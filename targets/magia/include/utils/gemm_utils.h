/*
 * Copyright (C) 2025 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * SPDX-License-Identifier: Apache-2.0
 *
 * MAGIA GEMM Utils — fp16 conversion helpers
 */

#ifndef GEMM_UTILS_H
#define GEMM_UTILS_H

#include <stdint.h>

/* Decode an fp16 bit pattern to a signed integer in millis (value × 1000).
 * Works directly on the bit string: sign(1)|exp(5)|mant(10).
 * Normal:    value = (1024+mant) × 2^(exp-25),  millis = (1024+mant)×1000 << or >> shift
 * Subnormal: value = mant × 2^(-24),             millis ≈ 0 for all representable values
 * Inf/NaN:   returns MAX to force an error */
static inline int32_t fp16_to_millis(uint16_t bits)
{
    int32_t sign = (bits & 0x8000) ? -1 : 1;
    int32_t exp  = (bits >> 10) & 0x1F;
    int32_t mant = bits & 0x3FF;

    if (exp == 31) /* inf / nan */
        return sign * 0x7FFFFFFF;
    if (exp == 0) /* subnormal or zero: value*1000 = mant*1000 >> 24, rounds to 0 */
        return sign * ((mant * 1000) >> 24);

    /* normal: value*1000 = (1024+mant) * 1000 * 2^(exp-25) */
    int32_t v     = (1024 + mant) * 1000;
    int32_t shift = exp - 25;
    return sign * (shift >= 0 ? v << shift : v >> -shift);
}

/* Convert float16 (IEEE 754 half-precision, 5-bit exp, 10-bit mant) bits
 * directly to double64 via bit manipulation, bypassing all soft-float helpers
 * (__extendohfdf2, __extendsfdf2) absent from this toolchain's libgcc.
 * Subnormals are flushed to zero for display. Used for printing. */
static inline double fp16_to_f64(uint16_t h)
{
    union {
        uint64_t u;
        double d;
    } u;
    uint64_t sign = (uint64_t)(h >> 15);
    uint64_t exp  = (uint64_t)((h >> 10) & 0x1f);
    uint64_t mant = (uint64_t)(h & 0x3ff);
    if (exp == 0)
        u.u = sign << 63;
    else
        u.u = (sign << 63) | ((exp - 15 + 1023) << 52) | (mant << 42);
    return u.d;
}

#endif /* GEMM_UTILS_H */
