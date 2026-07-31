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
    /* Ordered, not vfredusum: RVV leaves the unordered fold's association up to the
     * implementation, which makes the result unreproducible and so ungoldenable. The
     * networks that chain softmaxes (mobilevit's linear attention) need it to be exact,
     * and at 16..1000 lanes the ordered fold costs nothing worth measuring. */
    asm volatile ("vfredosum.vs v8, v0, v8");
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

/* The same fast exp as the vector path: COEF * x + BIAS in FP16, truncated to an
 * unsigned integer and reinterpreted as FP16 bits (vfcvt.rtz.xu.f.v saturates, so
 * the conversion is clamped here rather than left undefined). */
static inline _Float16 fastexp_scalar(_Float16 v)
{
    _Float16 t;
    _Float16 out;
    uint16_t bits;
    float f;

    t = (_Float16)(v * (_Float16)1486.0f);
    t = (_Float16)(t + (_Float16)15360.0f);

    f = (float)t;
    if (f <= 0.0f)
        bits = 0;
    else if (f >= 65535.0f)
        bits = 65535;
    else
        bits = (uint16_t)f;

    __builtin_memcpy(&out, &bits, sizeof(out));

    return out;
}

/* Row-at-a-time softmax matching the vector path step for step (max, fast exp of
 * x - max, ascending sum, then the per-element divide). Used for rows whose
 * vector accesses would not be 4-byte aligned. */
static void softmax_row_scalar(const _Float16 *src, _Float16 *dst, size_t len)
{
    _Float16 max;
    _Float16 sum;

    max = src[0];
    for (size_t i = 1; i < len; i++)
        if (src[i] > max)
            max = src[i];

    sum = 0.0f;
    for (size_t i = 0; i < len; i++) {
        _Float16 e = fastexp_scalar((_Float16)(src[i] - max));

        dst[i] = e;
        sum = (_Float16)(sum + e);
    }

    for (size_t i = 0; i < len; i++)
        dst[i] = (_Float16)(dst[i] / sum);
}

int softmax_fp16_spatz_task(void)
{
    volatile softmax_fp16_spatz_params_t *params;
    uintptr_t params_addr;
    _Float16 *shard_input;
    _Float16 *shard_output;
    _Float16 max;
    _Float16 sum;
    size_t r_len;
    size_t w_len;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile softmax_fp16_spatz_params_t *) params_addr;

    shard_input = (_Float16 *)params->shard_input;
    shard_output = (_Float16 *)params->shard_output;
    r_len = params->r_len;
    w_len = params->w_len;

    for (size_t r = 0; r < r_len; r++) {
        _Float16 *current_shard_input;
        _Float16 *current_shard_output;
        _Float16 max;
        _Float16 sum;

        current_shard_input = shard_input + (r * w_len);
        current_shard_output = shard_output + (r * w_len);

        /* The Spatz VLSU corrupts vector accesses to non-aligned addresses. All the
         * accesses of a row are unit-stride from its base, so checking the two row
         * bases is enough - with an odd w_len that is every other row. */
        if ((((uintptr_t)current_shard_input & 3u) == 0) &&
            (((uintptr_t)current_shard_output & 3u) == 0)) {
            max = find_max(current_shard_input, w_len);
            sum = compute_exponential_sum_fastexp(current_shard_input, current_shard_output, w_len, max);
            normalize(current_shard_output, w_len, sum);
        } else {
            softmax_row_scalar(current_shard_input, current_shard_output, w_len);
        }
    }

    return 0;
}
