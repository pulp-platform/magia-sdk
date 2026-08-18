// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// PULP cluster stage of the nn_is_bench block: row-wise softmax + argmax on the
// tile's post-LayerNorm block, in int16 packed SIMD.
//
// Programming model: pure SPMD, no inter-core communication. Core i owns rows
// [i*rows_per_core, (i+1)*rows_per_core) and its own result slot, so every core
// runs to its natural end and then writes PULP_DONE. ClusterRegs counts the
// NB_CORES_TO_WAIT completions and raises one event to the CV32, which is
// parked in WFE inside eu_pulp_wait() -- that hardware join *is* the barrier.
//
// Per row, six passes:
//   1. fp16 -> Q8.8 int16                           (scalar, bit manipulation)
//   2. max reduction                                 (pv.max.h)
//   3. subtract the max                              (pv.sub.h)
//   4. exp() of the non-positive residuals -> Q0.15  (scalar integer poly)
//   5. normalise by the reciprocal of the sum        (scalar)
//   6. sum of squared probabilities                  (pv.sdotsp.h)
//
// No floating point is used: the PULP cores only have zfinx/Xf16, and going
// through integer bit manipulation keeps this kernel independent of that.

#include "tile.h"
#include "nn_is_params.h"

/* ==========================================================================
 * Packed-SIMD helpers, two int16 lanes per 32-bit register.
 *
 * Written as inline asm because the mnemonic depends on the toolchain, not on
 * the hardware: both spellings assemble to the same encodings, which GVSoC
 * models in CoreV2.
 *
 *   compiler=GCC_MULTILIB -> riscv64-unknown-elf-gcc with xcvsimd, CORE-V
 *                            spelling `cv.*`; its assembler knows no `pv.*`.
 *   compiler=GCC_PULP     -> riscv32-unknown-elf-gcc with xgap9, PULP spelling
 *                            `pv.*`; its assembler knows no `cv.*`.
 *
 * CV32E40P is set to 1 for the multilib toolchain by the top-level CMakeLists
 * and reaches us through PULP_CFLAGS_DEFINES.
 * ========================================================================== */

#if defined(CV32E40P) && (CV32E40P == 1)
#define NN_OP_MAX2   "cv.max.h"
#define NN_OP_SUB2   "cv.sub.h"
#define NN_OP_SDOTSP "cv.sdotsp.h"
#else
#define NN_OP_MAX2   "pv.max.h"
#define NN_OP_SUB2   "pv.sub.h"
#define NN_OP_SDOTSP "pv.sdotsp.h"
#endif

/* The kernel walks the same int16 block both element-wise (int16_t *) and as
 * SIMD pairs (32-bit words). That is type punning, so the pair view has to be
 * may_alias: without it GCC's strict-aliasing analysis at -O2 is free to assume
 * the two views are disjoint and reorder the passes against each other. */
typedef uint32_t u32_pair __attribute__((may_alias));

/* rd = {max(a.hi, b.hi), max(a.lo, b.lo)}, signed */
static inline uint32_t simd_max2(uint32_t a, uint32_t b)
{
    uint32_t r;
    asm(NN_OP_MAX2 " %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

/* rd = {a.hi - b.hi, a.lo - b.lo} */
static inline uint32_t simd_sub2(uint32_t a, uint32_t b)
{
    uint32_t r;
    asm(NN_OP_SUB2 " %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

/* acc += a.lo*b.lo + a.hi*b.hi, signed, 32-bit accumulator */
static inline int32_t simd_sdotsp2(int32_t acc, uint32_t a, uint32_t b)
{
    asm(NN_OP_SDOTSP " %0, %1, %2" : "+r"(acc) : "r"(a), "r"(b));
    return acc;
}

/* ==========================================================================
 * Scalar integer helpers
 * ========================================================================== */

/* Saturation limit of the Q8.8 conversion: +/-32.0.
 *
 * Deliberately far below INT16_MAX. The SIMD subtract of the row maximum in
 * pass 3 computes q[i] - max with wrapping int16 arithmetic, so the operands
 * must leave headroom: clamping both sides to +/-8192 bounds the residual to
 * [-16384, 0]. Nothing is lost, because exp() already underflows Q0.15 at a
 * residual of about -11.1. */
#define Q88_SAT (8192)

/* fp16 bit pattern -> Q8.8 fixed point, saturating.
 *
 * value = (-1)^s * (1024 + m) * 2^(e - 25), so value * 256 is
 * (1024 + m) * 2^(e - 17). */
static inline int32_t f16bits_to_q88(uint16_t h)
{
    int32_t e = (h >> 10) & 0x1F;
    int32_t m = h & 0x3FF;
    int32_t v;

    if (e == 0) {
        return 0; /* zero or subnormal: below Q8.8 resolution */
    }
    if (e == 31) {
        v = Q88_SAT; /* inf / nan */
    } else {
        int32_t shift = e - 17;
        v             = 1024 + m;
        v             = (shift >= 0) ? (v << shift) : (v >> (-shift));
        if (v > Q88_SAT) {
            v = Q88_SAT;
        }
    }
    return (h & 0x8000) ? -v : v;
}

/* exp(d) for d <= 0 given in Q8.8, result in Q0.15 (32768 == 1.0).
 *
 * exp(d) = 2^(d*log2 e). Split the exponent into integer and fractional parts
 * and evaluate 2^-y on y in [0,1) with the quadratic interpolant through
 * y = 0, 1/2, 1:  2^-y ~= 1 - 0.6716*y + 0.1716*y^2  (error < 2e-3). */
static inline int32_t exp_neg_q15(int32_t d_q88)
{
    int32_t t, u, n, f, p;

    t = -d_q88;
    if (t <= 0) {
        return 32768; /* d == 0 -> the row maximum itself */
    }

    u = (t * 23637) >> 14; /* t * log2(e), still Q8.8 */
    n = u >> 8;
    if (n >= 16) {
        return 0; /* < 2^-16, invisible in Q0.15 */
    }
    f = u & 0xFF; /* fractional part, Q0.8 */

    p = 32768 - ((22006 * f) >> 8) + ((5624 * f * f) >> 16);

    return p >> n;
}

/* ==========================================================================
 * Task
 * ========================================================================== */

void nn_softmax_pulp_task(uint32_t data_ptr)
{
    volatile nn_pulp_params_t *params = (volatile nn_pulp_params_t *)data_ptr;

    const uint16_t *src = (const uint16_t *)params->src;
    int16_t *dst        = (int16_t *)params->dst;
    uint32_t cols       = params->cols;
    uint32_t rows_per   = params->rows_per_core;
    uint32_t core       = GET_PULP_LOCAL_ID(get_hartid());
    uint32_t row_first  = core * rows_per;
    uint32_t pairs      = cols >> 1;

    volatile nn_pulp_core_res_t *res = (volatile nn_pulp_core_res_t *)params->res + core;

    int32_t best_val   = -32768;
    uint32_t best_idx  = 0;
    uint32_t sumsq_acc = 0;
    uint32_t r;

    for (r = 0; r < rows_per; r++) {
        uint32_t row      = row_first + r;
        const uint16_t *s = src + (size_t)row * cols;
        int16_t *q        = dst + (size_t)row * cols;
        u32_pair *qp      = (u32_pair *)q;

        uint32_t acc, mxpair;
        int32_t mx, lo, hi, sum, recip, ss;
        uint32_t i, j;

        /* 1. fp16 -> Q8.8, staged in the output buffer */
        for (i = 0; i < cols; i++) {
            q[i] = (int16_t)f16bits_to_q88(s[i]);
        }

        /* 2. SIMD max reduction over the int16 pairs */
        acc = qp[0];
        for (j = 1; j < pairs; j++) {
            acc = simd_max2(acc, qp[j]);
        }
        lo = (int16_t)(acc & 0xFFFF);
        hi = (int16_t)(acc >> 16);
        mx = (lo > hi) ? lo : hi;

        /* argmax inside the row: first index reaching the maximum */
        for (i = 0; i < cols; i++) {
            if (q[i] == (int16_t)mx) {
                break;
            }
        }
        if (mx > best_val) {
            best_val = mx;
            best_idx = row * cols + i;
        }

        /* 3. SIMD subtract of the row maximum -> all lanes <= 0 */
        mxpair = ((uint32_t)(uint16_t)(int16_t)mx << 16) | (uint16_t)(int16_t)mx;
        for (j = 0; j < pairs; j++) {
            qp[j] = simd_sub2(qp[j], mxpair);
        }

        /* 4. exp of the residuals, accumulating the unclamped sum.
         * Only the row maximum can produce 32768, which is clamped to 32767 on
         * the way into int16 -- a 1-LSB effect on a single element. */
        sum = 0;
        for (i = 0; i < cols; i++) {
            int32_t e = exp_neg_q15(q[i]);
            sum += e;
            q[i] = (int16_t)(e > 32767 ? 32767 : e);
        }

        /* 5. normalise: out = e * 2^15 / sum, via one reciprocal per row.
         * e <= sum, so e*recip <= 2^30 and the product stays in int32. */
        recip = (sum > 0) ? (int32_t)((int32_t)(1 << 30) / sum) : 0;
        for (i = 0; i < cols; i++) {
            int32_t o = ((int32_t)q[i] * recip) >> 15;
            q[i]      = (int16_t)(o > 32767 ? 32767 : o);
        }

        /* 6. SIMD sum of squared probabilities.
         * Bounded over the whole row by max(p) * sum(p) = 2^15 * 2^15, so the
         * int32 accumulator cannot overflow; the >> 16 then keeps the cross-row
         * accumulator comfortably inside uint32. */
        ss = 0;
        for (j = 0; j < pairs; j++) {
            ss = simd_sdotsp2(ss, qp[j], qp[j]);
        }
        sumsq_acc += (uint32_t)ss >> 16;

        res->rows_done = r + 1;
    }

    res->argmax = best_idx;
    res->maxval = best_val;
    res->sumsq  = sumsq_acc;
}
