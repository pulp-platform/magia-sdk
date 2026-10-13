/*
 * GroupNorm, fp16, one shard of the groups per tile.
 *
 * The statistics are the delicate part. A group here can be as large as C*H*W (mobilevit
 * normalises with num_groups = 1 over 65536 elements), and at those sizes the obvious
 * formulation overflows FP16 several times over: the plain sum reaches ~1e6, the sum of
 * squares ~1.8e9, a single squared deviation ~5.8e6, and even the divisor - (_Float16)len
 * for len = 65536 - is Inf, since FP16 stops at 65504.
 *
 * So both reductions are scaled down *before* they accumulate, by
 *
 *     sh = the smallest power of two >= sqrt(len)
 *
 * chosen as a power of two so that dividing by it is exact and costs no precision at all:
 *
 *     mean = (sum of x/sh) * sh/len
 *     var  = (sum of ((x-mean)/sh)^2) * sh^2/len
 *
 * The accumulated quantities are then O(len/sh) and O(var*len/sh^2) - both a few thousand
 * for every shape this has been run on - and the two trailing rescalings happen once per
 * group in FP32, where var itself is allowed to exceed the FP16 range (it does: 73,242 on
 * one mobilevit layer) because only 1/sqrt(var + eps) is ever rounded back to FP16.
 *
 * The lane folds are ordered (vfredosum, not vfredsum). RVV leaves the unordered fold's
 * association to the implementation, so a golden cannot model it; ordered costs one pass
 * over 128 lanes per group.
 */
#include "tile.h"
#include "groupnorm_fp16_spatz_params.h"

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
 * Sum of x/sh over the group, folded ascending, then rescaled to the mean in FP32.
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

static inline void affine_channel(_Float16 *dst, const _Float16 gamma, const _Float16 beta, const size_t len)
{
    _Float16 *p_dst;
    size_t avl;
    size_t vl;

    p_dst = dst;
    avl = len;

    for (; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        asm volatile ("vle16.v v0, (%0)" :: "r"(p_dst));

        asm volatile ("vfmul.vf v0, v0, %0" :: "f"(gamma));
        asm volatile ("vfadd.vf v0, v0, %0" :: "f"(beta));

        asm volatile ("vse16.v v0, (%0)" :: "r"(p_dst));

        p_dst += vl;
    }
}

int groupnorm_fp16_spatz_task(void)
{
    volatile groupnorm_fp16_spatz_params_t *params;
    uintptr_t params_addr;

    _Float16 *gamma_base;
    _Float16 *beta_base;
    _Float16 *src_base;
    _Float16 *dst_base;
    _Float16 inv_sh;
    _Float16 denom;
    _Float16 eps;

    uint32_t elements_per_group;
    uint32_t g_len;
    uint32_t c_per_g;
    uint32_t hw_len;
    uint32_t g_start;
    uint32_t c_out;
    uint32_t sh;

    float sh_over_n;
    float sh2_over_n;
    float var;

    _Float16 mean;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile groupnorm_fp16_spatz_params_t *) params_addr;

    src_base   = (_Float16 *)params->shard_X;
    dst_base   = (_Float16 *)params->shard_Y;
    gamma_base = (_Float16 *)params->gamma;
    beta_base  = (_Float16 *)params->beta;
    eps        = *(_Float16 *)params->eps;

    g_start    = params->g_start;
    g_len      = params->g_len;
    c_per_g    = params->c_per_g;
    hw_len     = params->hw_len;
    c_out      = params->c_out;
    elements_per_group = c_per_g * hw_len;

    sh         = stat_scale(elements_per_group);
    inv_sh     = (_Float16)(1.0f / (float)sh);
    sh_over_n  = (float)sh / (float)elements_per_group;
    sh2_over_n = ((float)sh * (float)sh) / (float)elements_per_group;

    /* shard_X holds only this tile's groups, so the input and the output are indexed the
     * same way; only the gamma/beta lookup needs the global group number. */
    for (uint32_t g = 0; g < g_len; g++) {
        uint32_t group_offset = g * elements_per_group;
        uint32_t global_g = g_start + g;

        mean = compute_mean(src_base + group_offset, elements_per_group, inv_sh, sh_over_n);
        var  = compute_variance_scaled(src_base + group_offset, mean, inv_sh, elements_per_group)
               * sh2_over_n;

        denom = (_Float16)(1.0f / sqrtf_sp(var + (float)eps));

        normalize(src_base + group_offset, dst_base + group_offset, mean, denom, elements_per_group);

        for (uint32_t c = 0; c < c_per_g; c++) {
            uint32_t channel_offset = group_offset + (c * hw_len);
            uint32_t gamma_beta_idx = (global_g * c_per_g + c) % c_out;

            _Float16 g_val = gamma_base[gamma_beta_idx];
            _Float16 b_val = beta_base[gamma_beta_idx];

            affine_channel(dst_base + channel_offset, g_val, b_val, hw_len);
        }
    }

    return 0;
}
