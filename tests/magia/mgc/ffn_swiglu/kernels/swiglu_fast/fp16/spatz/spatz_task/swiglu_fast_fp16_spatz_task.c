#include "tile.h"
#include "utils/spatz_recip.h"
#include "swiglu_fast_fp16_spatz_params.h"

/* SiLU(g) * u = g * u / (1 + exp(-g)), all in FP16 (one rounding per instruction):
     e = exp(-g), the Schraudolph exp of the expminus kernels: the FP16 bits of
         rtz(max(g, -11) * -COEF + BIAS) (one fused multiply-add), 0 for g > ~10.4;
     d = 1 + e, in [1, 61000]: finite, so no clamp of the divisor;
     p = g * u;
     y = 1 / d: a = d * 2^-4 (exact, so that 1/a is normal), seed 0x77a0 - bits(a), 2 Newton steps
         y += y * (1 - a * y), then y * 2^-4;
     h = p / d: q0 = p * y, then one fused correction step q0 + y * (p - d * q0) (SPATZ_VFDIV_VV
         without the sign and range handling, for d >= 1). */
int swiglu_fast_fp16_spatz_task(void)
{
    const swiglu_fast_fp16_spatz_params_t *params;
    const _Float16 *G;
    const _Float16 *U;
    _Float16 *Y;
    size_t avl;
    size_t vl;

    SPATZ_F16_CONST(g_min, 0xC980); /* -11 */
    SPATZ_F16_CONST(coef, 0xE5C5);  /* -1477 ~ -1024 / ln(2) */
    SPATZ_F16_CONST(bias, 0x7380);  /* 15360 = 15 << 10 */
    SPATZ_F16_CONST(one, 0x3C00);   /* 1.0 */
    SPATZ_F16_CONST(scale, 0x2C00); /* 2^-4 */

    params = (const swiglu_fast_fp16_spatz_params_t *)mmio32(SPATZ_DATA);
    G      = (const _Float16 *)params->shard_G;
    U      = (const _Float16 *)params->shard_U;
    Y      = (_Float16 *)params->shard_Y;
    avl    = params->len;

    asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vfmv.v.f v24, %0" ::"f"(bias));

    for (; avl > 0; avl -= vl, G += vl, U += vl, Y += vl) {
        asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile("vle16.v v0, (%0)" ::"r"(G));
        asm volatile("vle16.v v4, (%0)" ::"r"(U));
        asm volatile(SPATZ_FPU_FENCE "vfmax.vf v8, v0, %0" ::"f"(g_min));
        asm volatile("vfmadd.vf v8, %0, v24" ::"f"(coef));  /* -g * COEF + BIAS */
        asm volatile(SPATZ_FPU_FENCE "vfcvt.rtz.xu.f.v v8, v8"); /* e = exp(-g) */
        asm volatile("vfadd.vf v8, v8, %0" ::"f"(one));     /* d = 1 + e */
        asm volatile("vfmul.vv v4, v0, v4");                 /* p = g * u */

        asm volatile("vfmul.vf v12, v8, %0" ::"f"(scale));   /* a = d * 2^-4 */
        asm volatile("vrsub.vx v16, v12, %0" ::"r"(0x77a0)); /* y ~ 1 / a */
        asm volatile("vfmv.v.f v20, %0" ::"f"(one));
        asm volatile("vfnmsac.vv v20, v12, v16");            /* 1 - a * y */
        asm volatile("vfmacc.vv v16, v16, v20");
        asm volatile("vfmv.v.f v20, %0" ::"f"(one));
        asm volatile("vfnmsac.vv v20, v12, v16");
        asm volatile("vfmacc.vv v16, v16, v20");
        asm volatile("vfmul.vf v16, v16, %0" ::"f"(scale));  /* y ~ 1 / d */

        asm volatile("vfmul.vv v12, v4, v16");               /* q0 = p * y */
        asm volatile("vfnmsub.vv v8, v12, v4");              /* p - d * q0 */
        asm volatile("vfmacc.vv v12, v16, v8");              /* h = q0 + y * (p - d * q0) */
        asm volatile("vse16.v v12, (%0)" SPATZ_VST_FENCE ::"r"(Y) : "memory");
    }

    return 0;
}
