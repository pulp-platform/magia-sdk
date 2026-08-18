// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Spatz task 2 of the nn_is_bench block: row-wise LayerNorm with an affine tail,
// in place over the tile's post-GEMM block.
//
//   for each row r:  mu   = mean(p[r])
//                    var  = mean((p[r] - mu)^2)
//                    p[r] = (p[r] - mu) / sqrt(var + eps) * gamma + beta
//
// Three vector passes per row. The reduction/sqrt idiom (vfredosum.vs +
// vfmv.f.s + fsqrt.h) is the one used by tests/spatz_on_magia/onnx_layernorm,
// already validated against the GVSoC Spatz model.

#include "tile.h"
#include "nn_is_params.h"

static inline _Float16 f16_from_bits(uint16_t bits)
{
    union {
        uint16_t u;
        _Float16 f;
    } c;
    c.u = bits;
    return c.f;
}

static inline _Float16 sqrt_hp(_Float16 x)
{
    _Float16 out;
    asm volatile("fsqrt.h %0, %1" : "=f"(out) : "f"(x));
    return out;
}

/* Sum of a row. v8 is the elementwise accumulator, v16 the reduction target. */
static _Float16 row_sum(const _Float16 *src, size_t len)
{
    register _Float16 ZERO asm("fs2") = (_Float16)0.0f;
    const _Float16 *p_src             = src;
    size_t avl                        = len;
    _Float16 sum;
    size_t vl;

    asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vfmv.v.f v8, %0" ::"f"(ZERO));
    asm volatile("vfmv.v.f v16, %0" ::"f"(ZERO));

    for (; avl > 0; avl -= vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile("vle16.v v0, (%0)" ::"r"(p_src));
        asm volatile("vfadd.vv v8, v8, v0");
        p_src += vl;
    }

    asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(len));
    asm volatile("vfredosum.vs v16, v8, v16");
    asm volatile("vfmv.f.s %0, v16" : "=f"(sum));

    return sum;
}

/* Sum of squared deviations from mu. */
static _Float16 row_sumsq(const _Float16 *src, _Float16 mu, size_t len)
{
    register _Float16 ZERO asm("fs2") = (_Float16)0.0f;
    register _Float16 MU asm("fs4")   = mu;
    const _Float16 *p_src             = src;
    size_t avl                        = len;
    _Float16 sum;
    size_t vl;

    asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vfmv.v.f v8, %0" ::"f"(ZERO));
    asm volatile("vfmv.v.f v16, %0" ::"f"(ZERO));

    for (; avl > 0; avl -= vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile("vle16.v v0, (%0)" ::"r"(p_src));
        asm volatile("vfsub.vf v0, v0, %0" ::"f"(MU));
        asm volatile("vfmul.vv v0, v0, v0");
        asm volatile("vfadd.vv v8, v8, v0");
        p_src += vl;
    }

    asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(len));
    asm volatile("vfredosum.vs v16, v8, v16");
    asm volatile("vfmv.f.s %0, v16" : "=f"(sum));

    return sum;
}

/* In-place (x - mu) * a + b over one row. */
static void row_affine(_Float16 *dst, _Float16 mu, _Float16 a, _Float16 b, size_t len)
{
    register _Float16 MU asm("fs4") = mu;
    register _Float16 A asm("fs5")  = a;
    register _Float16 B asm("fs6")  = b;
    _Float16 *p_dst                 = dst;
    size_t avl                      = len;
    size_t vl;

    for (; avl > 0; avl -= vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile("vle16.v v0, (%0)" ::"r"(p_dst));
        asm volatile("vfsub.vf v0, v0, %0" ::"f"(MU));
        asm volatile("vfmul.vf v0, v0, %0" ::"f"(A));
        asm volatile("vfadd.vf v0, v0, %0" ::"f"(B));
        asm volatile("vse16.v v0, (%0)" ::"r"(p_dst));
        p_dst += vl;
    }
}

int nn_ln_spatz_task(void)
{
    volatile nn_spatz_ln_params_t *params;
    uintptr_t params_addr;
    _Float16 *base;
    _Float16 gamma;
    _Float16 beta;
    _Float16 eps;
    _Float16 inv_cols;
    uint32_t rows;
    uint32_t cols;
    uint32_t r;

    params_addr = mmio32(SPATZ_DATA);
    params      = (volatile nn_spatz_ln_params_t *)params_addr;

    base     = (_Float16 *)params->src;
    rows     = params->rows;
    cols     = params->cols;
    gamma    = f16_from_bits(params->gamma);
    beta     = f16_from_bits(params->beta);
    eps      = f16_from_bits(params->eps);
    inv_cols = f16_from_bits(params->inv_cols);

    for (r = 0; r < rows; r++) {
        _Float16 *row = base + (size_t)r * cols;
        _Float16 mu   = row_sum(row, cols) * inv_cols;
        _Float16 var  = row_sumsq(row, mu, cols) * inv_cols;
        /* gamma / sqrt(var + eps), folded into a single vector multiplier */
        _Float16 a = gamma / sqrt_hp(var + eps);

        row_affine(row, mu, a, beta, cols);
    }

    return 0;
}
