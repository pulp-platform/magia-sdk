#include "tile.h"
#include "onnx_layernorm_params.h"

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

    mean = sum / len;

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

    var = sum / len;

    return var;
}

static inline void normalize(const _Float16 *src, _Float16 *dst, const _Float16 mean, const _Float16 var, const _Float16 eps, const size_t len)
{
    const _Float16 *p_src;
    _Float16 *p_dst;
    _Float16 denom;
    size_t avl;
    size_t vl;

    denom = 1 / sqrtf_hp(var + eps);
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
        p_dst +=vl;
    }
}

int onnx_layernorm_task(void)
{
    volatile onnx_layernorm_params_t *params;
    uintptr_t params_addr;
    _Float16 *gamma;
    _Float16 *beta;
    _Float16 *src;
    _Float16 *dst;
    _Float16 eps;
    size_t len;

    _Float16 mean;
    _Float16 var;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile onnx_layernorm_params_t *) params_addr;

    gamma = (_Float16 *)params->addr_gamma;
    beta = (_Float16 *)params->addr_beta;
    src = (_Float16 *)params->addr_src;
    dst = (_Float16 *)params->addr_res;
    eps = *(_Float16 *)params->addr_eps;
    len = params->len;

    mean = compute_mean(src, len);
    var = compute_variance(src, mean, len);

    normalize(src, dst, mean, var, eps, len);
    affine(dst, gamma, beta, len);

    return 0;
}
