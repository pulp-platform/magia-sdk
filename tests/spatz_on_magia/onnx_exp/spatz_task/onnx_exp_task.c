#include "tile.h"
#include "onnx_exp_params.h"

/**
 * Computes a fast vectorized approximation of exp(x)
 * Uses the bit-level fast exponential method (Schraudolph 1999):
 *      exp(x) ~ reinterpret_fp16(COEF * x + BIAS)
 * COEF = 2^mantissa / ln(2) (mantissa = 10 for FP16), scales x to the FP16 exponent range.
 * BIAS = exponent_bias * 2^mantissa (bias = 15 for FP16), shifts the bits to approximate exp.
 */
static inline void fast_exp(const _Float16 *src, _Float16 *dst, size_t len)
{
    register _Float16 MIN asm ("f8") = -10.0f;
    register _Float16 MAX asm ("f9") = +10.0f;
    register _Float16 COEF asm ("f10") = 1477.0f;
    register _Float16 BIAS asm ("f11") = 15360.0f;

    const _Float16 *p_src;
    _Float16 *p_dst;
    size_t avl;
    size_t vl;

    p_src = src;
    p_dst = dst;
    avl = len;

    for (; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        asm volatile ("vle16.v v0, (%0)" :: "r"(p_src));

        /* clamp for stability */
        asm volatile ("vfmin.vf v0, v0, %0" :: "f"(MAX));
        asm volatile ("vfmax.vf v0, v0, %0" :: "f"(MIN));

        asm volatile ("vfmul.vf v0, v0, %0" :: "f"(COEF));
        asm volatile ("vfadd.vf v0, v0, %0" :: "f"(BIAS));

        asm volatile ("vfcvt.rtz.xu.f.v v0, v0");

        asm volatile ("vse16.v v0, (%0)" :: "r"(p_dst));

        p_src += vl;
        p_dst += vl;
    }
}

int onnx_exp_task(void)
{
    volatile onnx_exp_params_t *params;
    uintptr_t params_addr;
    _Float16 *result;
    _Float16 *input;
    size_t len;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile onnx_exp_params_t *) params_addr;

    input = (_Float16 *)params->chunk_input;
    result = (_Float16 *)params->chunk_res;
    len = params->len;

    fast_exp(input, result, len);

    return 0;
}
