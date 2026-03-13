#include "tile.h"
#include "onnx_relu_params.h"

int onnx_relu_task(void)
{
    volatile onnx_relu_params_t *params;
    uintptr_t params_addr;

    _Float16 ZERO_f = 0.0f;
    _Float16 *src;
    _Float16 *dst;

    size_t len;
    size_t avl;
    size_t vl;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile onnx_relu_params_t *) params_addr;

    src = (_Float16 *)params->addr_src;
    dst = (_Float16 *)params->addr_res;
    len = params->len;

    avl = len;
    for (; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile ("vle16.v v0, (%0)" :: "r"(src));
        asm volatile ("vfmax.vf v8, v0, %0" :: "f"(ZERO_f));
        asm volatile ("vse16.v v8, (%0)" :: "r"(dst));

        src += vl;
        dst += vl;
    }

    return 0;
}
