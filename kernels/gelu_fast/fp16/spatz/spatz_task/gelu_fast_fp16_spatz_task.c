#include "tile.h"
#include "utils/spatz_recip.h"
#include "gelu_fast_fp16_spatz_params.h"

/* GELU(x), tanh form, x * sigmoid(2 * sqrt(2 / pi) * (x + 0.044715 * x^3)) = x / (1 + exp(-K * z)),
   z = x * (1 + 0.044715 * x^2), K = 2 * sqrt(2 / pi), all in FP16 (one rounding per instruction):
     z = x * (x * x * C3 + 1);
     e = exp(-K * z), the Schraudolph exp of swiglu_fast with K folded into COEF: the FP16 bits of
         rtz(min(max(z, -6.875), 16) * -COEF + BIAS) (one fused multiply-add), 0 for z > ~6.5; the upper
         clamp keeps the multiply-add finite (it would be -inf for z > ~34);
     d = 1 + e, in [1, 53000]: finite, so no clamp of the divisor;
     y = 1 / d: a = d * 2^-4 (exact, so that 1/a is normal), seed 0x77a0 - bits(a), 2 Newton steps
         y += y * (1 - a * y), then y * 2^-4;
     h = x / d: q0 = x * y, then one fused correction step q0 + y * (x - d * q0) (SPATZ_VFDIV_VV
         without the sign and range handling, for d >= 1). */
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
