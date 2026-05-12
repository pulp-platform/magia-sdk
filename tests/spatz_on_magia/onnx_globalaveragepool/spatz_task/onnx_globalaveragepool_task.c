#include "tile.h"
#include "onnx_globalaveragepool_params.h"

static inline void globalaveragepool(const _Float16 *src, _Float16 *dst, const uint32_t len)
{
    register _Float16 ZERO asm ("fs0") = 0.0f;
    const _Float16 *p_src;
    size_t original_avl;
    _Float16 sum;
    size_t avl;
    size_t vl;

    original_avl = len;
    p_src = src;
    avl = len;

    asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(len));
    asm volatile ("vfmv.v.f v0, %0" :: "f"(ZERO));
    asm volatile ("vfmv.v.f v8, %0" :: "f"(ZERO));

    for (; avl > 0; avl -=vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile ("vle16.v v16, (%0)" :: "r"(p_src));
        asm volatile ("vfadd.vv v0, v16, v0");

        p_src += vl;
    }

    asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(original_avl));
    asm volatile ("vfredosum.vs v8, v0, v8");
    asm volatile ("vfmv.f.s %0, v8" : "=f"(sum));

    *dst = sum / (_Float16)len;
}

int onnx_globalaveragepool_task(void)
{
    volatile onnx_globalaveragepool_params_t *params;
    uintptr_t params_addr;
    _Float16 *src;
    _Float16 *dst;
    size_t len;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile onnx_globalaveragepool_params_t *) params_addr;

    src = (_Float16 *) params->addr_input;
    dst = (_Float16 *) params->addr_res;
    len = params->len;

    globalaveragepool(src, dst, len);

    return 0;
}
