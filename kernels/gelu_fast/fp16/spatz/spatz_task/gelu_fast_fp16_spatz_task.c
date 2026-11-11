#include "tile.h"
#include "utils/spatz_recip.h"
#include "gelu_fast_fp16_spatz_params.h"

/*
 * GELU, tanh approximation, computed entirely in FP16
 * (one rounding per instruction).
 *
 *   GELU(x) = x * sigmoid(K * z)  =  x / (1 + exp(-K * z))
 *   where  z = x * (1 + C3 * x^2),  C3 = 0.044715,  K = 2 * sqrt(2 / pi)
 *
 * Steps:
 *
 * 1. z = x * (x * x * C3 + 1)
 *
 * 2. e = exp(-K * z), Schraudolph approximation.
 *    K is folded into COEF, so one fused multiply-add gives the FP16 bits of e:
 *        e_bits = rtz(clamp(z, -6.875, 16) * -COEF + BIAS)
 *    - The result is 0 for z > ~6.5.
 *    - The upper clamp (16) keeps the FMA finite; without it the FMA
 *      would overflow to -inf for z > ~34.
 *
 * 3. d = 1 + e
 *    d is in [1, ~59400], always finite, so the divisor needs no clamp.
 *
 * 4. y = 1 / d, by Newton-Raphson:
 *    - Scale: a = d * 2^-4 (exact). This keeps 1/a a normal number;
 *      1/d itself would be subnormal for large d.
 *    - Seed:  y = bits(0x77a0 - bits(a))
 *    - Twice: y += y * (1 - a * y)
 *    - Undo the scale: y = y * 2^-4
 *
 * 5. h = x / d:
 *    - q0 = x * y
 *    - One fused correction step: h = q0 + y * (x - d * q0)
 *    This is SPATZ_VFDIV_VV without its sign and range handling,
 *    which is not needed here because d >= 1.
 */
int gelu_fast_fp16_spatz_task(void)
{
    const gelu_fast_fp16_spatz_params_t *params;
    const _Float16 *X;
    _Float16 *Y;
    size_t avl;
    size_t vl;

    SPATZ_F16_CONST(z_min, 0xC6E0); /* -6.875 */
    SPATZ_F16_CONST(z_max, 0x4C00); /* 16 */
    SPATZ_F16_CONST(c3, 0x29B9);    /* 0.044715 */
    SPATZ_F16_CONST(coef, 0xE89A);  /* -2356 ~ -K * 1024 / ln(2) */
    SPATZ_F16_CONST(bias, 0x7380);  /* 15360 = 15 << 10 */
    SPATZ_F16_CONST(one, 0x3C00);   /* 1.0 */
    SPATZ_F16_CONST(scale, 0x2C00); /* 2^-4 */

    params = (const gelu_fast_fp16_spatz_params_t *)mmio32(SPATZ_DATA);
    X      = (const _Float16 *)params->shard_X;
    Y      = (_Float16 *)params->shard_Y;
    avl    = params->len;

    asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vfmv.v.f v24, %0" ::"f"(bias));
    asm volatile("vfmv.v.f v28, %0" ::"f"(one));

    for (; avl > 0; avl -= vl, X += vl, Y += vl) {
        asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile("vle16.v v0, (%0)" ::"r"(X));
        asm volatile(SPATZ_FPU_FENCE "vfmul.vv v4, v0, v0"); /* x * x */
        asm volatile("vfmadd.vf v4, %0, v28" ::"f"(c3));      /* x * x * C3 + 1 */
        asm volatile("vfmul.vv v4, v4, v0");                  /* z */
        asm volatile("vfmax.vf v8, v4, %0" ::"f"(z_min));
        asm volatile("vfmin.vf v8, v8, %0" ::"f"(z_max));
        asm volatile("vfmadd.vf v8, %0, v24" ::"f"(coef));    /* -K * z * COEF + BIAS */
        asm volatile(SPATZ_FPU_FENCE "vfcvt.rtz.xu.f.v v8, v8"); /* e = exp(-K * z) */
        asm volatile("vfadd.vf v8, v8, %0" ::"f"(one));       /* d = 1 + e */

        asm volatile("vfmul.vf v12, v8, %0" ::"f"(scale));    /* a = d * 2^-4 */
        asm volatile("vrsub.vx v16, v12, %0" ::"r"(0x77a0));  /* y ~ 1 / a */
        asm volatile("vfmv.v.f v20, %0" ::"f"(one));
        asm volatile("vfnmsac.vv v20, v12, v16");             /* 1 - a * y */
        asm volatile("vfmacc.vv v16, v16, v20");
        asm volatile("vfmv.v.f v20, %0" ::"f"(one));
        asm volatile("vfnmsac.vv v20, v12, v16");
        asm volatile("vfmacc.vv v16, v16, v20");
        asm volatile("vfmul.vf v16, v16, %0" ::"f"(scale));   /* y ~ 1 / d */

        asm volatile("vfmul.vv v12, v0, v16");                /* q0 = x * y */
        asm volatile("vfnmsub.vv v8, v12, v0");               /* x - d * q0 */
        asm volatile("vfmacc.vv v12, v16, v8");               /* h = q0 + y * (x - d * q0) */
        asm volatile("vse16.v v12, (%0)" SPATZ_VST_FENCE ::"r"(Y) : "memory");
    }

    return 0;
}
