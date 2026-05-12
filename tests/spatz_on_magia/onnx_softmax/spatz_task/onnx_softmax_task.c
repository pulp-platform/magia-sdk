#include "tile.h"
#include "onnx_softmax_params.h"

static inline _Float16 find_max(const _Float16 *vec, size_t len)
{
    const _Float16 *p_vec;
    _Float16 max;

    size_t original_avl;
    size_t avl;
    size_t vl;

    max = vec[0];
    p_vec = vec;

    original_avl = len;
    avl = len;

    asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile ("vfmv.s.f v0, %0" :: "f"(max));
    asm volatile ("vfmv.v.f v8, %0" :: "f"(max));

    for(; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile ("vle16.v v16, (%0)" :: "r"(p_vec));
        asm volatile ("vfmax.vv v8, v16, v8");

        p_vec += vl;
    }

    asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(original_avl));
    asm volatile ("vfredmax.vs v0, v8, v0");
    asm volatile ("vfmv.f.s %0, v0" : "=f"(max));

    return max;
}

/**
 * Computes a fast vectorized approximation of exp(x - max) for an FP16 vector
 * and returns the sum. Uses the bit-level fast exponential method (Schraudolph 1999):
 *      exp(x) ~ reinterpret_fp16(COEF * x + BIAS)
 * COEF = 2^mantissa / ln(2) (mantissa = 10 for FP16), scales x to the FP16 exponent range.
 * BIAS = exponent_bias * 2^mantissa (bias = 15 for FP16), shifts the bits to approximate exp.
 */
static inline _Float16 compute_exponential_sum_fastexp(const _Float16 *src, _Float16 *dst, size_t len, _Float16 max)
{
    register _Float16 COEF asm("f10") = 1486.0f;
    register _Float16 BIAS asm("f11") = 15360.0f;
    register _Float16 ZERO_f asm("f2") = 0.0f;

    const _Float16 *p_src;
    _Float16 *p_dst;
    _Float16 sum;

    size_t original_avl;
    size_t avl;
    size_t vl;

    p_src = src;
    p_dst = dst;

    original_avl = len;
    avl = len;
    sum = 0;

    asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile ("vfmv.v.f v0, %0" :: "f"(ZERO_f));
    asm volatile ("vfmv.v.f v8, %0" :: "f"(ZERO_f));

    for (; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        asm volatile ("vle16.v v16, (%0)" :: "r"(p_src));

        asm volatile ("vfsub.vf v16, v16, %0" :: "f"(max));

        asm volatile ("vfmul.vf v16, v16, %0" :: "f"(COEF));
        asm volatile ("vfadd.vf v16, v16, %0" :: "f"(BIAS));

        asm volatile ("vfcvt.rtz.xu.f.v v16, v16");

        asm volatile ("vse16.v v16, (%0)" :: "r"(p_dst));
        asm volatile ("vfadd.vv v0, v0, v16");

        p_src += vl;
        p_dst += vl;
    }

    asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(original_avl));
    asm volatile ("vfredusum.vs v8, v0, v8");
    asm volatile ("vfmv.f.s %0, v8" : "=f"(sum));

    return sum;
}

static inline void normalize(_Float16 *dst, size_t len, _Float16 sum)
{
    _Float16 *p_dst;
    size_t avl;
    size_t vl;

    p_dst = dst;
    avl = len;

    for (; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        asm volatile ("vle16.v v8, (%0)" :: "r"(p_dst));

        asm volatile ("vfdiv.vf v8, v8, %0" :: "f"(sum));

        asm volatile ("vse16.v v8, (%0)" :: "r"(p_dst));

        p_dst += vl;
    }
}

int onnx_softmax_task(void)
{
    volatile onnx_softmax_params_t *params;
    uintptr_t params_addr;
    _Float16 *src;
    _Float16 *dst;
    _Float16 max;
    _Float16 sum;
    size_t len;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile onnx_softmax_params_t *) params_addr;

    src = (_Float16 *)params->addr_src;
    dst = (_Float16 *)params->addr_res;
    len = params->len;

    max = find_max(src, len);
    sum = compute_exponential_sum_fastexp(src, dst, len, max);
    normalize(dst, len, sum);

    return 0;
}
