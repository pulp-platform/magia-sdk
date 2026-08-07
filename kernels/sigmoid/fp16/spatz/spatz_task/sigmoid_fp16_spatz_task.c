#include "tile.h"
#include "sigmoid_fp16_spatz_params.h"

/* Uncomment for a scalar, accurate sigmoid (range-reduced fp32 exp) instead of the vectorized
   Schraudolph fast exp. For debugging/verification only: correct but slow. */
// #define ACCURATE_SIGMOID

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

#ifdef ACCURATE_SIGMOID

static float exp_scalar(float x)
{
    /* exp(x) = 2^n * exp(r), with n = round(x / ln2) and r = x - n*ln2 in [-ln2/2, ln2/2]. */
    const float LOG2E = 1.4426950408889634f;
    const float LN2   = 0.6931471805599453f;

    float y = x * LOG2E;
    int n = (int) (y + (y >= 0.0f ? 0.5f : -0.5f));

    if (n <= -127)
        return 0.0f;   /* underflow: 2^n rounds to zero */

    float r = x - (float) n * LN2;

    /* exp(r) on the reduced range (6-term Taylor). */
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f + r * (0.041666668f + r * 0.008333334f))));

    /* 2^n by building the IEEE-754 single-precision exponent field. */
    union { float f; int i; } pow2 = { .i = (n + 127) << 23 };

    return p * pow2.f;
}

static void sigmoid_accurate(const _Float16 *src, _Float16 *dst, const size_t len)
{
    /* Scalar reference: numerically stable sigmoid e^min(x,0) / (1 + e^-|x|) with an accurate
       fp32 exp. Same math as the vectorized kernel, but exact enough to isolate the fast-exp error. */
    for (size_t i = 0; i < len; i++) {
        float x = (float) src[i];
        float a = x < 0.0f ? x : 0.0f;
        float b = x < 0.0f ? x : -x;

        dst[i] = (_Float16) (exp_scalar(a) / (1.0f + exp_scalar(b)));
    }
}

#endif

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

#ifdef ACCURATE_SIGMOID
    sigmoid_accurate(X, Y, len);
#else
    sigmoid(X, Y, len);
#endif
}
