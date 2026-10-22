#include "tile.h"
#include "utils/spatz_recip.h"
#include "instancenorm_fp16_spatz_params.h"

static inline _Float16 sqrtf_hp(_Float16 x)
{
    return spatz_sqrt_f16(x); /* no fsqrt on Spatz, see spatz_recip.h */
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

    p_src        = src;
    original_avl = len;
    avl          = len;

    asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vfmv.v.f v8, %0" ::"f"(ZERO));
    asm volatile("vfmv.v.f v16, %0" ::"f"(ZERO));

    for (; avl > 0; avl -= vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile("vle16.v v0, (%0)" ::"r"(p_src));
        asm volatile("vfadd.vv v8, v8, v0");
        p_src += vl;
    }

    asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(original_avl));
    asm volatile("vfredsum.vs v16, v8, v16");
    asm volatile("vfmv.f.s %0, v16" : "=f"(sum));

    mean = spatz_div_f16(sum, (_Float16)len);
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

    p_src        = src;
    original_avl = len;
    avl          = len;

    asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vfmv.v.f v8, %0" ::"f"(ZERO));
    asm volatile("vfmv.v.f v16, %0" ::"f"(ZERO));

    for (; avl > 0; avl -= vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile("vle16.v v0, (%0)" ::"r"(p_src));
        asm volatile("vfsub.vf v0, v0, %0" ::"f"(mean));
        asm volatile("vfmul.vv v0, v0, v0");
        asm volatile("vfadd.vv v8, v8, v0");
        p_src += vl;
    }

    asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(original_avl));
    asm volatile("vfredosum.vs v16, v8, v16");
    asm volatile("vfmv.f.s %0, v16" : "=f"(sum));

    var = spatz_div_f16(sum, (_Float16)len);
    return var;
}

static inline void normalize(const _Float16 *src,
                             _Float16 *dst,
                             const _Float16 mean,
                             const _Float16 var,
                             const _Float16 eps,
                             const size_t len)
{
    const _Float16 *p_src;
    _Float16 *p_dst;
    _Float16 denom;
    size_t avl;
    size_t vl;

    denom = spatz_div_f16((_Float16)1, sqrtf_hp(var + eps));
    p_src = src;
    p_dst = dst;
    avl   = len;

    for (; avl > 0; avl -= vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile("vle16.v v0, (%0)" ::"r"(p_src));
        asm volatile("vfsub.vf v0, v0, %0" ::"f"(mean));
        asm volatile("vfmul.vf v0, v0, %0" ::"f"(denom));
        asm volatile("vse16.v v0, (%0)" SPATZ_VST_FENCE ::"r"(p_dst));
        p_src += vl;
        p_dst += vl;
    }
}

static inline void
affine(_Float16 *dst, const _Float16 gamma, const _Float16 beta, const size_t len)
{
    _Float16 *p_dst;
    size_t avl;
    size_t vl;

    p_dst = dst;
    avl   = len;

    for (; avl > 0; avl -= vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile("vle16.v v0, (%0)" ::"r"(p_dst));
        asm volatile("vfmul.vf v0, v0, %0" ::"f"(gamma));
        asm volatile("vfadd.vf v0, v0, %0" ::"f"(beta));
        asm volatile("vse16.v v0, (%0)" SPATZ_VST_FENCE ::"r"(p_dst) : "memory");
        p_dst += vl;
    }
}

int instancenorm_fp16_spatz_task(void)
{
    volatile instancenorm_fp16_spatz_params_t *params;
    uintptr_t params_addr;

    const _Float16 *src;
    _Float16 *dst;

    const _Float16 *gamma;
    const _Float16 *beta;
    _Float16 epsilon;
    uint32_t inst_len;
    uint32_t hw_len;
    uint32_t num_channels;

    params_addr = mmio32(SPATZ_DATA);
    params      = (volatile instancenorm_fp16_spatz_params_t *)params_addr;

    src          = (_Float16 *)params->shard_input;
    dst          = (_Float16 *)params->shard_output;
    epsilon      = *(_Float16 *)params->eps;
    gamma        = (_Float16 *)params->gamma;
    beta         = (_Float16 *)params->beta;
    inst_len     = params->inst_len;
    hw_len       = params->hw_len;
    num_channels = params->num_channels;

    for (int i = 0; i < inst_len; i++) {
        uint32_t inst_off;
        uint32_t global_inst_idx;
        uint32_t c_idx;
        _Float16 gamma_c;
        _Float16 beta_c;
        _Float16 mean;
        _Float16 var;

        inst_off = i * hw_len;

        global_inst_idx = params->inst_start + i;
        c_idx           = global_inst_idx % num_channels;

        gamma_c = gamma[c_idx];
        beta_c  = beta[c_idx];

        mean = compute_mean(src + inst_off, hw_len);
        var  = compute_variance(src + inst_off, mean, hw_len);

        normalize(src + inst_off, dst + inst_off, mean, var, epsilon, hw_len);
        affine(dst + inst_off, gamma_c, beta_c, hw_len);
    }

    return 0;
}
