#include "tile.h"
#include "sigmoid_fp16_spatz_params.h"

static inline void sigmoid(const _Float16 *src, _Float16 *dst, const size_t len)
{
    register _Float16 BIAS asm ("fs0") = 15360.0f;
    register _Float16 COEF asm ("fs1") = 1477.0f;
    register _Float16 ZERO asm ("fs2") = 0.0f;
    register _Float16 ONE  asm ("fs3") = 1.0f;
    const _Float16 *p_src;
    _Float16 *p_dst;
    size_t avl;
    size_t vl;

    p_src = src;
    p_dst = dst;
    avl = len;

    for(; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        asm volatile ("vle16.v v0, (%0)" :: "r"(p_src));

        /* Numerically stable sigmoid e^min(x,0) / (1 + e^-|x|): both exponents have a
           non-positive argument, so the fast exp approximation never overflows. */
        asm volatile ("vfmin.vf v8, v0, %0" :: "f"(ZERO));      /* v8  = min(x, 0) */
        asm volatile ("vfsgnjn.vv v24, v0, v0");                /* v24 = -x        */
        asm volatile ("vfmin.vv v16, v0, v24");                 /* v16 = -|x|      */

        /* numerator = fast_exp(min(x, 0)) */
        asm volatile ("vfmul.vf v8, v8, %0" :: "f"(COEF));
        asm volatile ("vfadd.vf v8, v8, %0" :: "f"(BIAS));
        asm volatile ("vfcvt.rtz.xu.f.v v8, v8");

        /* denominator = 1 + fast_exp(-|x|) */
        asm volatile ("vfmul.vf v16, v16, %0" :: "f"(COEF));
        asm volatile ("vfadd.vf v16, v16, %0" :: "f"(BIAS));
        asm volatile ("vfcvt.rtz.xu.f.v v16, v16");
        asm volatile ("vfadd.vf v16, v16, %0" :: "f"(ONE));

        asm volatile ("vfdiv.vv v0, v8, v16");                  /* sigmoid = num / den */

        asm volatile ("vse16.v v0, (%0)" :: "r"(p_dst));

        p_src += vl;
        p_dst += vl;
    }
}

int sigmoid_fp16_spatz_task(void)
{
    volatile sigmoid_fp16_spatz_params_t *params;
    uintptr_t params_addr;
    _Float16 *X;
    _Float16 *Y;
    size_t len;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile sigmoid_fp16_spatz_params_t *) params_addr;

    X = (_Float16 *) params->shard_X;
    Y = (_Float16 *) params->shard_Y;
    len = params->len;

    sigmoid(X, Y, len);
}
