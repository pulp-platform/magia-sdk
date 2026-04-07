#include "tile.h"
#include "onnx_hardswish_params.h"

static inline void hardswish(const _Float16 *src, _Float16 *dst, const size_t len)
{
    register _Float16 ALPHA asm ("fs0") = (1.0f / 6.0f);
    register _Float16 BETA asm ("fs1") = 0.5f;
    register _Float16 ZERO asm ("fs2") = 0.0f;
    register _Float16 ONE  asm ("fs3") = 1.0f;
    const _Float16 *p_src;
    _Float16 *p_dst;
    size_t avl;
    size_t vl;

    p_src = src;
    p_dst = dst;
    avl = len;

    for(; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        asm volatile ("vle16.v v0,( %0)" :: "r"(p_src));

        asm volatile ("vfmul.vf v8, v0, %0" :: "f"(ALPHA));
        asm volatile ("vfadd.vf v8, v8, %0" :: "f"(BETA));

        asm volatile ("vfmin.vf v8, v8, %0" :: "f"(ONE));
        asm volatile ("vfmax.vf v8, v8, %0" :: "f"(ZERO));
        asm volatile ("vfmul.vv v0, v8, v0");

        asm volatile ("vse16.v v0, (%0)" :: "r"(p_dst));

        p_src += vl;
        p_dst += vl;
    }
}

int onnx_hardswish_task(void)
{
    volatile onnx_hardswish_params_t *params;
    uintptr_t params_addr;
    _Float16 *src;
    _Float16 *dst;
    size_t len;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile onnx_hardswish_params_t *) params_addr;

    src = (_Float16 *)params->addr_input;
    dst = (_Float16 *)params->addr_res;
    len = params->len;

    hardswish(src, dst, len);

    return 0;
}
