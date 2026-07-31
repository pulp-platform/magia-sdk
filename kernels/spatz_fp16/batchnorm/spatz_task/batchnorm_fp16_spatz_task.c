#include "tile.h"
#include "batchnorm_fp16_spatz_params.h"

static inline void batchnorm(const _Float16 *src, const _Float16 mean, const _Float16 var, const _Float16 gamma, const _Float16 beta, const _Float16 epsilon, _Float16 *dst, const size_t len)
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

        /* centering: x_centered[i] = x[i] - mean[i] */
        asm volatile ("vfsub.vf v0, v0, %0" :: "f"(mean));

        /* standard deviation: std[i] = sqrt(var[i] + eps) */
        asm volatile ("vfmv.v.f v8, %0" :: "f"(var));
        asm volatile ("vfadd.vf v8, v8, %0" :: "f"(epsilon));
        asm volatile ("vfsqrt.v v8, v8");

        /* normalize: norm[i] = x_centered[i] / std[i] */
        asm volatile ("vfdiv.vv v0, v0, v8");

        /* affine: dst = norm[i] * gamma[i] + beta[i] */
        asm volatile ("vfmul.vf v0, v0, %0" :: "f"(gamma));
        asm volatile ("vfadd.vf v0, v0, %0" :: "f"(beta));

        asm volatile ("vse16.v v0, (%0)" :: "r"(p_dst) : "memory");

        p_src += vl;
        p_dst += vl;
    }
}

int batchnorm_fp16_spatz_task(void)
{
    volatile batchnorm_fp16_spatz_params_t *params;
    uintptr_t params_addr;

    const _Float16 *src;
    _Float16 *dst;

    const _Float16 *gamma;
    const _Float16 *beta;
    const _Float16 *mean;
    const _Float16 *var;
    _Float16 epsilon;
    uint32_t c_len;
    uint32_t input_C;
    uint32_t hw_len;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile batchnorm_fp16_spatz_params_t *) params_addr;

    src = (_Float16 *) params->shard_X;
    dst = (_Float16 *) params->shard_Y;
    epsilon = *(_Float16 *) params->eps;
    gamma = (_Float16 *) params->gamma;
    beta  = (_Float16 *) params->beta;
    mean  = (_Float16 *) params->mean;
    var   = (_Float16 *) params->var;
    c_len = params->c_len;
    hw_len = params->hw_len;
    input_C = params->channels;

    for (uint32_t c = 0; c < c_len; c++) {
        uint32_t c_off;
        uint32_t c_idx;
        _Float16 gamma_c;
        _Float16 beta_c;
        _Float16 mean_c;
        _Float16 var_c;

        c_off = c * hw_len;
        c_idx = (params->c_start + c) % input_C;

        gamma_c = gamma[c_idx];
        beta_c  = beta[c_idx];
        mean_c  = mean[c_idx];
        var_c   = var[c_idx];

        batchnorm(src + c_off, mean_c, var_c, gamma_c, beta_c, epsilon, dst + c_off, hw_len);
    }

    return 0;
}
