// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "tile.h"
#include "softmax_rows_fp16_spatz_params.h"

/* Unit-stride vector accesses must be 4-byte aligned. */
#define MISALIGNED(p) (((uintptr_t)(p)) & 0x2)

/* Row-wise softmax in place: X[r, :] = exp(s * (X[r, :] - max)) / sum, with the fast exp and
   the scale s folded into its clamp and coefficient (exact for s a power of 2), the sum
   accumulated in fp32 (one partial per lane, then an ordered reduction) and normalized by one
   fp16 reciprocal per row. */
int softmax_rows_fp16_spatz_task(void)
{
    const softmax_rows_fp16_spatz_params_t *params;
    register _Float16 bias asm("f11") = SOFTMAX_ROWS_EXP_BIAS;
    union { uint32_t u; _Float16 f[2]; } min_coef;
    _Float16 *X;
    _Float16 *p;
    _Float16 min, coef, m, rcp;
    float sum;
    size_t rows, cols, avl, vl, vl0;

    params     = (const softmax_rows_fp16_spatz_params_t *)mmio32(SPATZ_DATA);
    X          = (_Float16 *)params->shard_X;
    rows       = params->dims & 0xffff;
    cols       = params->dims >> 16;
    min_coef.u = params->min_coef;
    min        = min_coef.f[0];
    coef       = min_coef.f[1];

    if (MISALIGNED(X) || (cols & 1) || cols == 0)
        return 1;

    for (size_t r = 0; r < rows; r++, X += cols) {
        /* row max */
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl0) : "r"(cols));
        asm volatile("vle16.v v8, (%0)" ::"r"(X));
        for (p = X + vl0, avl = cols - vl0; avl > 0; avl -= vl, p += vl) {
            asm volatile("vsetvli %0, %1, e16, m8, tu, ma" : "=r"(vl) : "r"(avl));
            asm volatile("vle16.v v0, (%0)" ::"r"(p));
            asm volatile("vfmax.vv v8, v8, v0");
        }
        asm volatile("vsetvli zero, %0, e16, m8, ta, ma" ::"r"(vl0));
        asm volatile("vfredmax.vs v24, v8, v8");
        asm volatile("vfmv.f.s %0, v24" : "=f"(m));

        /* exp(s * (x - max)) in place, and its sum in fp32 */
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl0) : "r"(cols));
        asm volatile("vfmv.v.f v16, %0" ::"f"(0.0f));
        for (p = X, avl = cols; avl > 0; avl -= vl, p += vl) {
            asm volatile("vsetvli %0, %1, e16, m4, tu, ma" : "=r"(vl) : "r"(avl));
            asm volatile("vle16.v v0, (%0)" ::"r"(p));
            asm volatile("vfsub.vf v0, v0, %0" ::"f"(m));
            asm volatile("vfmax.vf v0, v0, %0" ::"f"(min));
            asm volatile("vfmul.vf v0, v0, %0" ::"f"(coef));
            asm volatile("vfadd.vf v0, v0, %0" ::"f"(bias));
            asm volatile("vfcvt.rtz.xu.f.v v0, v0");
            asm volatile("vse16.v v0, (%0)" ::"r"(p) : "memory");
            asm volatile("vfwadd.wv v16, v16, v0");
        }
        asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(vl0));
        asm volatile("vfmv.s.f v24, %0" ::"f"(0.0f));
        asm volatile("vfredusum.vs v24, v16, v24");
        asm volatile("vfmv.f.s %0, v24" : "=f"(sum));

        /* normalize */
        rcp = (_Float16)(1.0f / sum);
        for (p = X, avl = cols; avl > 0; avl -= vl, p += vl) {
            asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
            asm volatile("vle16.v v0, (%0)" ::"r"(p));
            asm volatile("vfmul.vf v0, v0, %0" ::"f"(rcp));
            asm volatile("vse16.v v0, (%0)" ::"r"(p) : "memory");
        }
    }

    return 0;
}
