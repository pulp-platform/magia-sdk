#include "tile.h"
#include "groupnorm_fp16_spatz_params.h"

static inline _Float16 sqrtf_hp(_Float16 x)
{
    _Float16 out;
    asm volatile ("fsqrt.h %0, %1" : "=f"(out) : "f"(x));
    return out;
}

static inline _Float16 compute_mean(const _Float16 *src, const size_t len)
{
    _Float16 ZERO = 0.0f;

    const _Float16 *p_src;
    size_t original_avl;
    _Float16 mean;
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
        asm volatile ("vfadd.vv v8, v8, v0");

        p_src += vl;
    }

    asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(original_avl));
    asm volatile ("vfredsum.vs v16, v8, v16");
    asm volatile ("vfmv.f.s %0, v16" : "=f"(sum));

    mean = sum / (_Float16)len;

    return mean;
}

static inline _Float16 compute_variance(const _Float16 *src, const _Float16 mean, const size_t len)
{
    _Float16 ZERO = 0.0f;

    const _Float16 *p_src;
    size_t original_avl;
    _Float16 var;
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
        asm volatile ("vfmul.vv v0, v0, v0");
        asm volatile ("vfadd.vv v8, v8, v0");

        p_src += vl;
    }

    asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(original_avl));
    asm volatile ("vfredosum.vs v16, v8, v16");
    asm volatile ("vfmv.f.s %0, v16" : "=f"(sum));

    var = sum / (_Float16)len;

    return var;
}

static inline void normalize(const _Float16 *src, _Float16 *dst, const _Float16 mean, const _Float16 var, const _Float16 eps, const size_t len)
{
    const _Float16 *p_src;
    _Float16 *p_dst;
    _Float16 denom;
    size_t avl;
    size_t vl;

    denom = 1.0f / sqrtf_hp(var + eps);
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
    _Float16 eps;

    uint32_t elements_per_group;
    uint32_t g_len;
    uint32_t c_per_g;
    uint32_t hw_len;
    uint32_t g_start;

    _Float16 mean;
    _Float16 var;

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
    elements_per_group = c_per_g * hw_len;

    for (uint32_t g = 0; g < g_len; g++) {
        uint32_t group_offset = g * elements_per_group;

        mean = compute_mean(src_base + group_offset, elements_per_group);
        var  = compute_variance(src_base + group_offset, mean, elements_per_group);

        normalize(src_base + group_offset, dst_base + group_offset, mean, var, eps, elements_per_group);

        for (uint32_t c = 0; c < c_per_g; c++) {
            uint32_t channel_offset = group_offset + (c * hw_len);
            uint32_t gamma_beta_idx = (g_start + g) * c_per_g + c;

            _Float16 g_val = gamma_base[gamma_beta_idx];
            _Float16 b_val = beta_base[gamma_beta_idx];

            affine_channel(dst_base + channel_offset, g_val, b_val, hw_len);
        }
    }

    return 0;
}
