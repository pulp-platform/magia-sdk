#include "tile.h"
#include "swiglu_fp16_spatz_params.h"

/* SiLU(g) * u in fp32, one rounding to fp16 at the end. SiLU(g) = g / (1 + exp(-g)): exp(-g) = 2^t,
   t = -g * log2(e), clamped to [-125, 126] (exp(-g) only has to be finite and normal); t = n + f with n = round(t), f in [-0.5, 0.5]; 2^f by a degree-5 polynomial (relative
   error < 2e-7), 2^n added to its exponent field. */
int swiglu_fp16_spatz_task(void)
{
    volatile swiglu_fp16_spatz_params_t *params;
    const _Float16 *G;
    const _Float16 *U;
    _Float16 *Y;
    size_t avl;
    size_t vl;

    _Float16 ZERO_H = 0.0f;
    float NLOG2E    = -1.44269504f;
    float T_MAX     = 126.0f;
    float T_MIN     = -125.0f;
    float C5        = 1.33335581e-3f;
    float C4        = 9.61812911e-3f;
    float C3        = 5.55041087e-2f;
    float C2        = 2.40226507e-1f;
    float C1        = 6.93147181e-1f;
    float ONE       = 1.0f;

    params = (volatile swiglu_fp16_spatz_params_t *)mmio32(SPATZ_DATA);
    G      = (const _Float16 *)params->shard_G;
    U      = (const _Float16 *)params->shard_U;
    Y      = (_Float16 *)params->shard_Y;
    avl    = params->elem_len;

    for (; avl > 0; avl -= vl) {
        asm volatile("vsetvli %0, %1, e16, m2, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile("vle16.v v0, (%0)" ::"r"(G));
        asm volatile("vle16.v v2, (%0)" ::"r"(U));
        asm volatile("vfwadd.vf v4, v0, %0" ::"f"(ZERO_H));  /* g, fp32 */
        asm volatile("vfwadd.vf v24, v2, %0" ::"f"(ZERO_H)); /* u, fp32 */

        asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"(vl));
        asm volatile("vfmul.vf v8, v4, %0" ::"f"(NLOG2E)); /* t = -g * log2(e) */
        asm volatile("vfmin.vf v8, v8, %0" ::"f"(T_MAX));
        asm volatile("vfmax.vf v8, v8, %0" ::"f"(T_MIN));
        asm volatile("vfcvt.x.f.v v12, v8");     /* n = round(t) */
        asm volatile("vfcvt.f.x.v v16, v12");
        asm volatile("vfsub.vv v16, v8, v16");   /* f = t - n */
        asm volatile("vfmul.vf v20, v16, %0" ::"f"(C5));
        asm volatile("vfadd.vf v20, v20, %0" ::"f"(C4));
        asm volatile("vfmul.vv v20, v20, v16");
        asm volatile("vfadd.vf v20, v20, %0" ::"f"(C3));
        asm volatile("vfmul.vv v20, v20, v16");
        asm volatile("vfadd.vf v20, v20, %0" ::"f"(C2));
        asm volatile("vfmul.vv v20, v20, v16");
        asm volatile("vfadd.vf v20, v20, %0" ::"f"(C1));
        asm volatile("vfmul.vv v20, v20, v16");
        asm volatile("vfadd.vf v20, v20, %0" ::"f"(ONE)); /* 2^f */
        asm volatile("vsll.vi v12, v12, 23");
        asm volatile("vadd.vv v20, v20, v12");             /* exp(-g) = 2^f * 2^n */
        asm volatile("vfadd.vf v20, v20, %0" ::"f"(ONE));
        asm volatile("vfdiv.vv v4, v4, v20");              /* g / (1 + exp(-g)) */
        asm volatile("vfmul.vv v4, v4, v24");              /* * u */

        asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(vl));
        asm volatile("vfncvt.f.f.w v0, v4");
        asm volatile("vse16.v v0, (%0)" ::"r"(Y) : "memory");

        G += vl;
        U += vl;
        Y += vl;
    }

    return 0;
}
