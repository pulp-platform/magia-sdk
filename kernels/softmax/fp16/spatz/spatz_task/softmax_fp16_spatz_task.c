#include "tile.h"
#include "softmax_fp16_spatz_params.h"

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

/**
 * Softmax over an inner (non-last) axis: the reduce axis has stride inner_dim, so the max, the
 * exponential sum and the division are all per-column vectors over inner_dim (vectorized over
 * inner_dim, looping over the reduce axis). exp(x - max) uses the Schraudolph fast-exp. Unlike the
 * inner_dim == 1 case, the three phases live in a single function: here max, sum and normalize
 * would each have to return a per-column vector, so fusing them keeps those vectors in registers
 * and avoids allocating extra buffers.
 */
static inline void softmax_strided(const _Float16 *src, _Float16 *dst, const size_t reduce_dim, const size_t inner_dim)
{
    register _Float16 COEF asm("f10") = 1486.0f;
    register _Float16 BIAS asm("f11") = 15360.0f;

    size_t avl = inner_dim;
    size_t vl;

    for (; avl > 0; avl -= vl) {
        size_t off = inner_dim - avl;
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        /* Column-wise max over the reduce axis (v0 holds the per-column max) */
        asm volatile ("vle16.v v0, (%0)" :: "r"(src + off));
        for (size_t r = 1; r < reduce_dim; r++) {
            asm volatile ("vle16.v v8, (%0)" :: "r"(src + (r * inner_dim) + off));
            asm volatile ("vfmax.vv v0, v0, v8");
        }

        /* dst = fastexp(src - max); v16 accumulates the per-column sum */
        asm volatile ("vmv.v.i v16, 0");
        for (size_t r = 0; r < reduce_dim; r++) {
            const _Float16 *p_src = src + (r * inner_dim) + off;
            _Float16 *p_dst = dst + (r * inner_dim) + off;

            asm volatile ("vle16.v v8, (%0)" :: "r"(p_src));
            asm volatile ("vfsub.vv v8, v8, v0");
            asm volatile ("vfmul.vf v8, v8, %0" :: "f"(COEF));
            asm volatile ("vfadd.vf v8, v8, %0" :: "f"(BIAS));
            asm volatile ("vfcvt.rtz.xu.f.v v8, v8");
            asm volatile ("vse16.v v8, (%0)" :: "r"(p_dst) : "memory");
            asm volatile ("vfadd.vv v16, v16, v8");
        }

        /* Normalize each row by the per-column sum */
        for (size_t r = 0; r < reduce_dim; r++) {
            _Float16 *p_dst = dst + (r * inner_dim) + off;

            asm volatile ("vle16.v v8, (%0)" :: "r"(p_dst));
            asm volatile ("vfdiv.vv v8, v8, v16");
            asm volatile ("vse16.v v8, (%0)" :: "r"(p_dst) : "memory");
        }
    }
}

int softmax_fp16_spatz_task(void)
{
    volatile softmax_fp16_spatz_params_t *params;
    uintptr_t params_addr;
    _Float16 *shard_input;
    _Float16 *shard_output;
    size_t reduce_dim;
    size_t inner_dim;
    size_t outer_len;
    size_t block;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile softmax_fp16_spatz_params_t *) params_addr;

    shard_input = (_Float16 *)params->shard_input;
    shard_output = (_Float16 *)params->shard_output;
    reduce_dim = params->reduce_dim;
    inner_dim = params->inner_dim;
    outer_len = params->outer_len;
    block = reduce_dim * inner_dim;

    for (size_t o = 0; o < outer_len; o++) {
        _Float16 *current_shard_input;
        _Float16 *current_shard_output;

        current_shard_input = shard_input + (o * block);
        current_shard_output = shard_output + (o * block);

        if (inner_dim == 1) {
            _Float16 max = find_max(current_shard_input, reduce_dim);
            _Float16 sum = compute_exponential_sum_fastexp(current_shard_input, current_shard_output, reduce_dim, max);
            normalize(current_shard_output, reduce_dim, sum);
        } else {
            softmax_strided(current_shard_input, current_shard_output, reduce_dim, inner_dim);
        }
    }

    return 0;
}
