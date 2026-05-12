#include "tile.h"
#include "onnx_batchnorm_params.h"

static inline void batchnorm(const _Float16 *src, const _Float16 *mean, const _Float16 *var, const _Float16 *gamma, const _Float16 *beta, const _Float16 epsilon, _Float16 *dst, const size_t len)
{
    const _Float16 *p_gamma;
    const _Float16 *p_beta;
    const _Float16 *p_mean;
    const _Float16 *p_var;
    const _Float16 *p_src;
    _Float16 *p_dst;
    size_t avl;
    size_t vl;

    p_gamma = gamma;
    p_beta = beta;
    p_mean = mean;
    p_var = var;
    p_src = src;
    p_dst = dst;
    avl = len;

    for (; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        asm volatile ("vle16.v v0, (%0)" :: "r"(p_src));
        asm volatile ("vle16.v v8, (%0)" :: "r"(p_mean));
        asm volatile ("vle16.v v16, (%0)" :: "r"(p_var));

        /* centering: x_centered[i] = x[i] - mean[i] */
        asm volatile ("vfsub.vv v0, v0, v8");

        /* standard deviation: std[i] = sqrt(var[i] + eps) */
        asm volatile ("vfadd.vf v16, v16, %0" :: "f"(epsilon));
        asm volatile ("vfsqrt.v v24, v16");

        /* normalize: norm[i] = x_centered[i] / std[i] */
        asm volatile ("vfdiv.vv v0, v0, v24");

        /* affine: dst = norm[i] * gamma[i] + beta[i] */
        asm volatile ("vle16.v v8, (%0)" :: "r"(p_gamma));
        asm volatile ("vle16.v v16, (%0)" :: "r"(p_beta));
        asm volatile ("vfmul.vv v0, v0, v8");
        asm volatile ("vfadd.vv v0, v0, v16");

        asm volatile ("vse16.v v0, (%0)" :: "r"(p_dst) : "memory");

        p_gamma += vl;
        p_mean += vl;
        p_beta += vl;
        p_var += vl;
        p_src += vl;
        p_dst += vl;
    }

}

// TODO: remove me :)
static inline _Float16 sqrtf_hp(_Float16 x)
{
    _Float16 out;
    asm volatile ("fsqrt.h %0, %1" : "=f"(out) : "f"(x));
    return out;
}

int onnx_batchnorm_task(void)
{
    volatile onnx_batchnorm_params_t *params;
    uintptr_t params_addr;
    _Float16 *gamma;
    _Float16 *beta;
    _Float16 *mean;
    _Float16 *var;
    _Float16 *src;
    _Float16 *dst;
    _Float16 eps;
    size_t len;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile onnx_batchnorm_params_t *) params_addr;

    gamma = (_Float16 *)params->addr_gamma;
    beta = (_Float16 *)params->addr_beta;
    mean = (_Float16 *)params->addr_mean;
    var = (_Float16 *)params->addr_var;
    src = (_Float16 *)params->addr_src;
    dst = (_Float16 *)params->addr_res;
    eps = *(_Float16 *)params->addr_eps;
    len = params->len;

    batchnorm(src, mean, var, gamma, beta, eps, dst, len);

    return 0;
}
