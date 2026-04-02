/*
 * Copyright (C) 2023-2025 ETH Zurich and University of Bologna
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
 * Authors: Alberto Dequino <alberto.dequino@unibo.it>
 *
 * MAGIA Attention ISA Utils
 */

#ifndef ATTENTION_UTILS_H
#define ATTENTION_UTILS_H

#include "magia_tile_utils.h"
#include <stdbool.h>

/* Compare two float16 values directly via their bit patterns.
 * IEEE 754 fp16 is monotonic within each sign, so integer comparison
 * on the raw bits works with sign handling.  Handles +0 == -0. */
static inline bool fp16_gt(float16 a, float16 b)
{
    uint16_t ua = *(uint16_t *)&a;
    uint16_t ub = *(uint16_t *)&b;
    if ((ua & 0x8000) != (ub & 0x8000)) {
        if ((ua | ub) & 0x7FFF)
            return !!(ub & 0x8000); /* positive > negative */
        return false;               /* +0 == -0            */
    }
    if (ua & 0x8000)
        return ua < ub; /* both negative: smaller magnitude wins */
    return ua > ub;     /* both positive: larger magnitude wins  */
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

/**
 * Element-wise comparison of the max vectors.
 * Saves in the curr buffer the bigger values.
 */
int max_compare(uint32_t curr, uint32_t prev, uint32_t dim)
{
    for (uint32_t i = 0; i < dim; i++) {
        float16 p = *(volatile float16 *)(prev + (i * 2));
        if (fp16_gt(p, *(volatile float16 *)(curr + (i * 2))))
            (*(volatile float16 *)(curr + (i * 2))) = p;
    }
}

/**
 * Finds the max value for each row and saves it in the result in the maxes buffer.
 */
int row_max(uint32_t s, uint32_t maxes, uint32_t dim_h, uint32_t dim_w)
{
    for (uint32_t i = 0; i < dim_h; i++) {
        uint32_t row  = s + i * dim_w * 2;
        float16 r_max = 0;

        for (uint32_t j = 0; j < dim_w; j++) {
            float16 val = *(volatile float16 *)(row + j * 2);
            if (fp16_gt(val, r_max))
                r_max = val;
        }
        (*(volatile float16 *)(maxes + i * 2)) = r_max;
    }
}

/**
 * For each row i of the input h x w matrix "s", substract the i-th element of the "m" vector.
 */
int row_diff(uint32_t s, uint32_t m, uint32_t h, uint32_t w)
{
    for (uint32_t i = 0; i < h; i++) {
        uint32_t row = s + i * w * 2;
        float16 diff = *(volatile float16 *)(m + i * 2);
        for (uint32_t j = 0; j < w; j++) {
            (*(volatile float16 *)(row + j * 2)) = (*(volatile float16 *)(row + j * 2)) - diff;
        }
    }
}

/**
 * For each row i of the input h x w matrix "s", sum the values and store it in the i-th element of
 * the "l" vector.
 */
int row_sum(uint32_t s, uint32_t l, uint32_t h, uint32_t w)
{
    for (uint32_t i = 0; i < h; i++) {
        uint32_t row = s + i * 2 * w;
        float16 sum  = 0;
        for (uint32_t j = 0; j < w; j++) {
            sum = sum + *(volatile float16 *)(row + j * 2);
        }
        (*(volatile float16 *)(l + i * 2)) = sum;
    }
}

/**
 * For each row i of the input h x w matrix "s", divide the values by the i-th element of the "m"
 * vector.
 */
int rowdiv(uint32_t s, uint32_t m, uint32_t h, uint32_t w)
{
    for (uint32_t i = 0; i < h; i++) {
        uint32_t row = s + i * w * 2;
        float16 div  = *(volatile float16 *)(m + i * 2);
        for (uint32_t j = 0; j < w; j++) {
            (*(volatile float16 *)(row + j * 2)) = (*(volatile float16 *)(row + j * 2)) / div;
        }
    }
}

/**
 * Element wise sum of v2 into v1
 */
int vect_sum(uint32_t v1, uint32_t v2, uint32_t dim)
{
    for (uint32_t i = 0; i < dim; i++) {
        (*(volatile float16 *)(v1 + i * 2)) =
            *(volatile float16 *)(v1 + i * 2) + *(volatile float16 *)(v2 + i * 2);
    }
}

/**
 * Element wise diff of v2 into v1
 */
int vect_diff(uint32_t v1, uint32_t v2, uint32_t dim)
{
    for (uint32_t i = 0; i < dim; i++) {
        (*(volatile float16 *)(v1 + i * 2)) =
            *(volatile float16 *)(v1 + i * 2) - *(volatile float16 *)(v2 + i * 2);
    }
}

/**
 * Element wise product of v2 into v1
 */
int vect_prod(uint32_t v1, uint32_t v2, uint32_t dim)
{
    for (uint32_t i = 0; i < dim; i++) {
        (*(volatile float16 *)(v1 + i * 2)) =
            (*(volatile float16 *)(v1 + i * 2)) * (*(volatile float16 *)(v2 + i * 2));
    }
}

// #define GIST_A  12102203.17133801f
// #define GIST_B  1064986823.010288f
// #define GIST_C  8388608
// #define GIST_D  2139095040

// float16 fastexp_gist(float16 x) {
//     x = GIST_A * x + GIST_B;

//     if (x < GIST_C || x > GIST_D)
//         x = (x < GIST_C) ? 0.0f : GIST_D;

//     uint32_t n = (uint32_t)(x);
//     return *(float16 *) &n;
// }

// fp16!
volatile float soft_expf(float x)
{
    if (x > 11.0f)
        x = 11.0f;
    if (x < -17.0f)
        return 0.0f;

    // Range reduction: x = n*ln2 + r, exp(x) = 2^n * exp(r)
    int n   = (int)(x * 1.4426950408889634f + (x >= 0 ? 0.5f : -0.5f));
    float r = x - (float)n * 0.6931471805599453f;

    // Taylor expansion of exp(r) for |r| <= ln2/2
    float e =
        1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f + r * (0.04166667f + r * 0.00833333f))));

    // Multiply by 2^n
    if (n > 0) {
        for (int i = 0; i < n; i++)
            e *= 2.0f;
    } else {
        for (int i = 0; i < -n; i++)
            e *= 0.5f;
    }

    return e;
}

int exponential(uint32_t matrix, uint32_t rows, uint32_t columns)
{
    for (uint32_t i = 0; i < rows; i++) {
        for (uint32_t j = 0; j < columns; j++) {
            volatile float16 *ptr = (volatile float16 *)(matrix + i * columns * 2 + j * 2);
            *ptr                  = (float16)soft_expf((float)(*ptr));
        }
    }
}

#endif // ATTENTION_UTILS_H
