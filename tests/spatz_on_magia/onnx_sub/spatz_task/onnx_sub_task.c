#include "tile.h"
#include "onnx_sub_params.h"

int onnx_sub_task(void)
{
    volatile onnx_sub_params_t *params;
    uintptr_t params_addr;
    _Float16 *src_a;
    _Float16 *src_b;
    _Float16 *dst;
    size_t avl;
    size_t vl;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile onnx_sub_params_t *) params_addr;

    src_a = (_Float16 *)params->addr_input_a;
    src_b = (_Float16 *)params->addr_input_b;
    dst = (_Float16 *)params->addr_res;
    avl = params->len;

    for (; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        asm volatile ("vle16.v v0, (%0)" :: "r"(src_a));
        asm volatile ("vle16.v v8, (%0)" :: "r"(src_b));

        asm volatile ("vfsub.vv v16, v0, v8");

        asm volatile ("vse16.v v16, (%0)" :: "r"(dst));

        src_a += vl;
        src_b += vl;
        dst += vl;
    }

    return 0;
}
