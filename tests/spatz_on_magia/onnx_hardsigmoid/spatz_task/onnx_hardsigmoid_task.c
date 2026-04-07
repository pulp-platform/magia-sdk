#include "tile.h"
#include "onnx_hardsigmoid_params.h"

static inline void hardsigmoid(const _Float16 *src, _Float16 *dst, const _Float16 alpha, const _Float16 beta, const size_t len)
{
    register _Float16 ZERO asm ("fs0") = 0.0f;
    register _Float16 ONE  asm ("fs1") = 1.0f;
    const _Float16 *p_src;
    _Float16 *p_dst;
    size_t avl;
    size_t vl;

    p_src = src;
    p_dst = dst;
    avl = len;

    asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile ("vfmv.v.f v8, %0" :: "f"(ONE));

    for(; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        asm volatile ("vle16.v v0,( %0)" :: "r"(p_src));

        asm volatile ("vfmul.vf v0, v0, %0" :: "f"(alpha));
        asm volatile ("vfadd.vf v0, v0, %0" :: "f"(beta));

        asm volatile ("vfmin.vf v0, v0, %0" :: "f"(ONE));
        asm volatile ("vfmax.vf v0, v0, %0" :: "f"(ZERO));

        asm volatile ("vse16.v v0, (%0)" :: "r"(p_dst));

        p_src += vl;
        p_dst += vl;
    }
}

int onnx_hardsigmoid_task(void)
{
    volatile onnx_hardsigmoid_params_t *params;
    uintptr_t params_addr;
    _Float16 alpha;
    _Float16 beta;
    _Float16 *src;
    _Float16 *dst;
    size_t len;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile onnx_hardsigmoid_params_t *) params_addr;

    alpha = *(_Float16 *)params->addr_alpha;
    beta = *(_Float16 *)params->addr_beta;
    src = (_Float16 *)params->addr_input;
    dst = (_Float16 *)params->addr_res;
    len = params->len;

    hardsigmoid(src, dst, alpha, beta, len);

    return 0;
}
