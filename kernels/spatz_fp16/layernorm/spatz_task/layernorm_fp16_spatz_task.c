/*
 * LayerNorm, fp16, one shard of the rows per tile.
 *
 * The statistics are taken the same way groupnorm takes them, and for the same reasons:
 * accumulating a row's sum and sum of squares straight in FP16 runs out of range long
 * before the shapes stop being reasonable, and the divisor - (_Float16)len - eventually
 * becomes Inf on its own. So both reductions are scaled down *before* they accumulate, by
 *
 *     sh = the smallest power of two >= sqrt(len)
 *
 * chosen as a power of two so that dividing by it is exact and costs no precision at all:
 *
 *     mean = (sum of x/sh) * sh/len
 *     var  = (sum of ((x-mean)/sh)^2) * sh^2/len
 *
 * The two trailing rescalings happen once per row in FP32, where the variance is allowed
 * to exceed the FP16 range because only 1/sqrt(var + eps) is ever rounded back to FP16.
 *
 * Both lane folds are ordered (vfredosum, not vfredsum). RVV leaves the unordered fold's
 * association to the implementation, so a golden cannot model it - compute_mean used to
 * fold with vfredsum while compute_variance folded with vfredosum, which made nothing
 * downstream of a LayerNorm reproducible.
 *
 * The affine is per element of the row (a gamma and a beta the length of the row), which
 * is the one real difference from groupnorm's per-channel broadcast.
 */
#include "tile.h"
#include "layernorm_fp16_spatz_params.h"

static inline float sqrtf_sp(float x)
{
    float out;
    asm volatile ("fsqrt.s %0, %1" : "=f"(out) : "f"(x));
    return out;
}

/* Smallest power of two >= sqrt(len), the exact scale the reductions are taken in. */
static inline uint32_t stat_scale(uint32_t len)
{
    uint32_t sh = 1;

    while (sh < 32768u && sh * sh < len)
        sh <<= 1;

    return sh;
}

/*
 * Sum of x/sh over the row, folded ascending, then rescaled to the mean in FP32.
 * inv_sh is 1/sh and so exactly representable; the per-element scaling is lossless.
 */
static inline _Float16 compute_mean(const _Float16 *src, const size_t len, const _Float16 inv_sh, const float sh_over_n)
{
    _Float16 ZERO = 0.0f;

    const _Float16 *p_src;
    size_t original_avl;
    _Float16 sum;
    size_t avl;
    size_t vl;

    p_src = src;
    original_avl = len;
    avl = len;

    asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile ("vfmv.v.f v8, %0" :: "f"(ZERO));
    asm volatile ("vfmv.v.f v16, %0" :: "f"(ZERO));

    for (; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        asm volatile ("vle16.v v0, (%0)" :: "r"(p_src));

        asm volatile ("vfmul.vf v0, v0, %0" :: "f"(inv_sh));
        asm volatile ("vfadd.vv v8, v8, v0");

        p_src += vl;
    }

    asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(original_avl));
    asm volatile ("vfredosum.vs v16, v8, v16");
    asm volatile ("vfmv.f.s %0, v16" : "=f"(sum));

    return (_Float16)((float)sum * sh_over_n);
}

/*
 * Sum of ((x - mean)/sh)^2, left unscaled - the caller multiplies by sh^2/len in FP32,
 * because the variance itself does not always fit FP16.
 */
static inline float compute_variance_scaled(const _Float16 *src, const _Float16 mean, const _Float16 inv_sh, const size_t len)
{
    _Float16 ZERO = 0.0f;

    const _Float16 *p_src;
    size_t original_avl;
    _Float16 sum;
    size_t avl;
    size_t vl;

    p_src = src;
    original_avl = len;
    avl = len;

    asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile ("vfmv.v.f v8, %0" :: "f"(ZERO));
    asm volatile ("vfmv.v.f v16, %0" :: "f"(ZERO));

    for (; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        asm volatile ("vle16.v v0, (%0)" :: "r"(p_src));

        asm volatile ("vfsub.vf v0, v0, %0" :: "f"(mean));
        asm volatile ("vfmul.vf v0, v0, %0" :: "f"(inv_sh));
        asm volatile ("vfmul.vv v0, v0, v0");
        asm volatile ("vfadd.vv v8, v8, v0");

        p_src += vl;
    }

    asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(original_avl));
    asm volatile ("vfredosum.vs v16, v8, v16");
    asm volatile ("vfmv.f.s %0, v16" : "=f"(sum));

    return (float)sum;
}

static inline void normalize(const _Float16 *src, _Float16 *dst, const _Float16 mean, const _Float16 denom, const size_t len)
{
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

        asm volatile ("vfsub.vf v0, v0, %0" :: "f"(mean));
        asm volatile ("vfmul.vf v0, v0, %0" :: "f"(denom));

        asm volatile ("vse16.v v0, (%0)" :: "r"(p_dst));

        p_src += vl;
        p_dst += vl;
    }
}

static inline void affine(_Float16 *dst, const _Float16 *gamma, const _Float16 *beta, const size_t len)
{
    const _Float16 *p_gamma;
    const _Float16 *p_beta;
    _Float16 *p_dst;
    size_t avl;
    size_t vl;

    p_gamma = gamma;
    p_beta = beta;
    p_dst = dst;
    avl = len;

    for (; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        asm volatile ("vle16.v v0, (%0)" :: "r"(p_dst));
        asm volatile ("vle16.v v8, (%0)" :: "r"(p_gamma));
        asm volatile ("vle16.v v16, (%0)" :: "r"(p_beta));

        asm volatile ("vfmul.vv v0, v0, v8");
        asm volatile ("vfadd.vv v0, v0, v16");

        asm volatile ("vse16.v v0, (%0)" :: "r"(p_dst));

        p_gamma += vl;
        p_beta += vl;
        p_dst += vl;
    }
}

int layernorm_fp16_spatz_task(void)
{
    volatile layernorm_fp16_spatz_params_t *params;
    uintptr_t params_addr;
    _Float16 *gamma_base;
    _Float16 *beta_base;
    _Float16 *shard_X;
    _Float16 *shard_Y;
    _Float16 inv_sh;
    _Float16 denom;
    _Float16 mean;
    _Float16 eps;
    size_t r_len;
    size_t w_len;
    uint32_t sh;
    float sh_over_n;
    float sh2_over_n;
    float var;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile layernorm_fp16_spatz_params_t *) params_addr;

    shard_X        = (_Float16 *)params->shard_X;
    shard_Y        = (_Float16 *)params->shard_Y;
    gamma_base     = (_Float16 *)params->gamma;
    beta_base      = (_Float16 *)params->beta;
    eps            = *(_Float16 *)params->eps;

    r_len          = params->r_len;
    w_len          = params->w_len;

    if (r_len == 0 || w_len == 0)
        return 0;

    sh         = stat_scale((uint32_t)w_len);
    inv_sh     = (_Float16)(1.0f / (float)sh);
    sh_over_n  = (float)sh / (float)w_len;
    sh2_over_n = ((float)sh * (float)sh) / (float)w_len;

    for (size_t r = 0; r < r_len; r++) {
        _Float16 *current_src;
        _Float16 *current_dst;

        current_src = shard_X + (r * w_len);
        current_dst = shard_Y + (r * w_len);

        mean = compute_mean(current_src, w_len, inv_sh, sh_over_n);
        var  = compute_variance_scaled(current_src, mean, inv_sh, w_len) * sh2_over_n;

        denom = (_Float16)(1.0f / sqrtf_sp(var + (float)eps));

        normalize(current_src, current_dst, mean, denom, w_len);
        affine(current_dst, gamma_base, beta_base, w_len);
    }

    return 0;
}
