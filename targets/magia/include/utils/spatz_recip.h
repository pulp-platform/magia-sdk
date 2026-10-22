// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <francesco.conti@chips.it>
//
// FP16 division for Spatz tasks without a divider (Spatz decodes no vfdiv,
// and MAGIA builds it without the DIVSQRT unit). A division is a reciprocal (integer-trick seed
// + Newton-Raphson steps), a multiplication and one correction step:
//   q0 = n * y;  r = n - d * q0 (fused);  q = q0 + r * y (fused)
// Within 1 ulp of the correctly rounded FP16 quotient: in a numpy model of the instruction
// sequence (one FP16 rounding per instruction) ~6 in 100000 random quotients are 1 ulp off, those
// lying within a hair of a rounding midpoint; bit-exact division would need the last step in FP32.
//
//   spatz_recip_f16/f32(d)               scalar 1/d (FP32 Newton), for SPATZ_VFDIV_VF
//   spatz_div_f16/f32(n, d)              scalar n / d, replaces C `/` (compiled to fdiv.h/fdiv.s)
//   spatz_sqrt_f16/f32(x)                scalar sqrt (FP32 Newton on 1/sqrt), replaces fsqrt.h/.s
//   SPATZ_VFDIV_VF(q, n, a, d, y)        vq = vn / d for a scalar d, y = spatz_recip_f16(d); va scratch
//   SPATZ_VFDIV_VV(q, n, d, a, y)        FP16 vq = vn / vd; va, vy scratch
//   SPATZ_VFDIV_VV_F32(q, n, d, a, y)    FP32 (e32) vq = vn / vd; va, vy scratch
//
// Register arguments are strings ("v8"), groups at the current LMUL, pairwise distinct; vn and vd
// are preserved. SPATZ_VFDIV_VV is within 1 ulp for every normal |d| in [2^-10, 65504]; for
// smaller |d| it divides by 2^-10, for |d| = inf by 65504 (q ~ n * 1.5e-5 instead of 0).

#ifndef SPATZ_RECIP_H
#define SPATZ_RECIP_H

#include <stdint.h>

/* 1 / x in FP32 with FP32 multiply-adds only: seed from the bit pattern, 3 Newton steps. The fused
   ops are asm: __builtin_fmaf is a libcall here (no libm in the Spatz tasks). */
static inline float spatz_recip_f32(float x)
{
    union {
        float f;
        uint32_t u;
    } ax, y;
    const float one = 1.0f;
    float e;

    ax.f = __builtin_fabsf(x);
    y.u  = 0x7EF311C3u - ax.u;
    for (int i = 0; i < 3; i++) {
        asm volatile("fnmsub.s %0, %1, %2, %3" : "=f"(e) : "f"(ax.f), "f"(y.f), "f"(one));
        asm volatile("fmadd.s %0, %1, %2, %1" : "=f"(y.f) : "f"(y.f), "f"(e));
    }
    return __builtin_copysignf(y.f, x);
}

static inline _Float16 spatz_recip_f16(_Float16 x)
{
    return (_Float16)spatz_recip_f32((float)x);
}

/* Scalar n / d without fdiv: q0 = n * (1/d), then one fused correction step. Equal to the correctly
   rounded quotient on 500000 random FP32 operands, and (via FP32) on 500000 FP16 ones. Use it for
   C-level divisions in Spatz tasks: the compiler turns `/` into fdiv.h/fdiv.s, also illegal. */
static inline float spatz_div_f32(float n, float d)
{
    const float y = spatz_recip_f32(d);
    float q       = n * y;
    float r;

    asm volatile("fnmsub.s %0, %1, %2, %3" : "=f"(r) : "f"(d), "f"(q), "f"(n));
    asm volatile("fmadd.s %0, %1, %2, %3" : "=f"(q) : "f"(r), "f"(y), "f"(q));
    return q;
}

static inline _Float16 spatz_div_f16(_Float16 n, _Float16 d)
{
    return (_Float16)spatz_div_f32((float)n, (float)d);
}

/* sqrt(x) in FP32 for x >= 0 without a sqrt unit: y = 1/sqrt(x) from the bit-pattern seed and 4
   Newton steps y *= 1.5 - 0.5*x*y*y, s = x*y, then one fused correction s += (x - s*s) * y/2.
   Equal to the correctly rounded FP32 sqrt on 500000 random operands, and rounded to FP16 to the
   correctly rounded FP16 sqrt for every non-negative FP16 x (numpy model). */
static inline float spatz_sqrt_f32(float x)
{
    union {
        float f;
        uint32_t u;
    } y;
    const float half_x     = 0.5f * x;
    const float three_half = 1.5f;
    float t, s, r, h;

    if (x == 0.0f)
        return x; /* the seed of 0 overflows y*y */
    y.f = x;
    y.u = 0x5F3759DFu - (y.u >> 1);
    for (int i = 0; i < 4; i++) {
        t = y.f * y.f;
        asm volatile("fnmsub.s %0, %1, %2, %3" : "=f"(t) : "f"(half_x), "f"(t), "f"(three_half));
        y.f = y.f * t;
    }
    s = x * y.f;
    h = 0.5f * y.f;
    asm volatile("fnmsub.s %0, %1, %2, %3" : "=f"(r) : "f"(s), "f"(s), "f"(x));
    asm volatile("fmadd.s %0, %1, %2, %3" : "=f"(s) : "f"(r), "f"(h), "f"(s));
    return s;
}

static inline _Float16 spatz_sqrt_f16(_Float16 x)
{
    return (_Float16)spatz_sqrt_f32((float)x);
}

/* FP16 constant for an "f" asm operand, materialized from its bits: clang 15 cannot allocate an FP
   register for a constant-folded _Float16 asm input ("couldn't allocate input reg for constraint
   'f'"), which is why the kernels pin theirs with `register _Float16 ... asm("fsN")`. */
#define SPATZ_F16_CONST(name, bits)                                                                \
    _Float16 name;                                                                                 \
    asm volatile("fmv.h.x %0, %1" : "=f"(name) : "r"(bits))

/* vq = vn / d for a scalar d whose reciprocal is y: q0 = n*y; r = n - d*q0; q = q0 + r*y */
#define SPATZ_VFDIV_VF(q, n, a, d, y)                                                              \
    do {                                                                                           \
        asm volatile("vfmul.vf " q ", " n ", %0" ::"f"(y));                                        \
        asm volatile("vfmul.vf " a ", " n ", %0" ::"f"(y));                                        \
        asm volatile("vfnmsub.vf " a ", %0, " n ::"f"(d));                                         \
        asm volatile("vfmacc.vf " q ", %0, " a ::"f"(y));                                          \
    } while (0)

/* vq = vn / vd. va = clamp(|vd| * 2^-4, 2^-14, 4094), vy = 1/va (seed 0x77a0 - bits(va), 2 Newton
   steps y += y * (1 - a*y) with a fused error term in vq), vy = sign(vd) * vy * 2^-4; the correction step uses the divisor the
   reciprocal was computed for, va = sign(vd) * va * 16 (equal to vd in range, finite for inf). */
#define SPATZ_VFDIV_VV(q, n, d, a, y)                                                              \
    do {                                                                                           \
        SPATZ_F16_CONST(_scale, 0x2C00); /* 2^-4 */                                                \
        SPATZ_F16_CONST(_unscl, 0x4C00); /* 2^4 */                                                 \
        SPATZ_F16_CONST(_lo, 0x0400);    /* 2^-14 */                                               \
        SPATZ_F16_CONST(_hi, 0x6BFF);    /* 4094 = 65504 * 2^-4 */                                 \
        SPATZ_F16_CONST(_one, 0x3C00);   /* 1.0 */                                                 \
        asm volatile(SPATZ_FPU_FENCE "vfsgnjx.vv " a ", " d ", " d);                               \
        asm volatile("vfmul.vf " a ", " a ", %0" ::"f"(_scale));                                   \
        asm volatile(SPATZ_FPU_FENCE "vfmax.vf " a ", " a ", %0" ::"f"(_lo));                      \
        asm volatile("vfmin.vf " a ", " a ", %0" ::"f"(_hi));                                      \
        asm volatile("vrsub.vx " y ", " a ", %0" ::"r"(0x77a0));                                   \
        asm volatile("vfmv.v.f " q ", %0" ::"f"(_one));                                            \
        asm volatile("vfnmsac.vv " q ", " a ", " y);                                               \
        asm volatile("vfmacc.vv " y ", " y ", " q);                                                \
        asm volatile("vfmv.v.f " q ", %0" ::"f"(_one));                                            \
        asm volatile("vfnmsac.vv " q ", " a ", " y);                                               \
        asm volatile("vfmacc.vv " y ", " y ", " q);                                                \
        asm volatile("vfmul.vf " a ", " a ", %0" ::"f"(_unscl));                                   \
        asm volatile(SPATZ_FPU_FENCE "vfsgnj.vv " a ", " a ", " d);                                \
        asm volatile("vfmul.vf " y ", " y ", %0" ::"f"(_scale));                                   \
        asm volatile(SPATZ_FPU_FENCE "vfsgnj.vv " y ", " y ", " d);                                \
        asm volatile("vfmul.vv " q ", " n ", " y);                                                 \
        asm volatile("vfnmsub.vv " a ", " q ", " n);                                               \
        asm volatile("vfmacc.vv " q ", " y ", " a);                                                \
    } while (0)

/* FP32 (e32) vq = vn / vd for normal, finite vd: va = |vd|, vy = 1/va (seed 0x7EF311C3 - bits(va),
   3 Newton steps with a fused error term in vq), sign, then the quotient with one correction step.
   Equal to the correctly rounded FP32 quotient on 400000 random silu operands (numpy model). */
#define SPATZ_VFDIV_VV_F32(q, n, d, a, y)                                                          \
    do {                                                                                           \
        const float _one = 1.0f;                                                                   \
        asm volatile(SPATZ_FPU_FENCE "vfsgnjx.vv " a ", " d ", " d);                               \
        asm volatile("vrsub.vx " y ", " a ", %0" ::"r"(0x7EF311C3));                               \
        for (int _i = 0; _i < 3; _i++) {                                                           \
            asm volatile("vfmv.v.f " q ", %0" ::"f"(_one));                                        \
            asm volatile("vfnmsac.vv " q ", " a ", " y);                                           \
            asm volatile("vfmacc.vv " y ", " y ", " q);                                            \
        }                                                                                          \
        asm volatile(SPATZ_FPU_FENCE "vfsgnj.vv " y ", " y ", " d);                                \
        asm volatile("vfmul.vv " q ", " n ", " y);                                                 \
        asm volatile(SPATZ_FPU_FENCE "vfsgnj.vv " a ", " a ", " d);                                \
        asm volatile("vfnmsub.vv " a ", " q ", " n);                                               \
        asm volatile("vfmacc.vv " q ", " y ", " a);                                                \
    } while (0)

#endif // SPATZ_RECIP_H
