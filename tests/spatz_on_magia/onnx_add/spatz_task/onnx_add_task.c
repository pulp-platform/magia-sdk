#include "magia_spatz_utils.h"
#include "magia_tile_utils.h"
#include "onnx_add_params.h"

int onnx_add_task(void)
{
    volatile onnx_add_params_t *params;
    uintptr_t params_addr;
    _Float16 *a;
    _Float16 *b;
    _Float16 *r;
    size_t avl;
    size_t vl;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile onnx_add_params_t *) params_addr;

    r = (_Float16 *)params->addr_res;
    a = (_Float16 *)params->addr_a;
    b = (_Float16 *)params->addr_b;
    avl = params->len;

    for (; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        asm volatile ("vle16.v v0, (%0)" :: "r"(a));
        asm volatile ("vle16.v v8, (%0)" :: "r"(b));

        asm volatile ("vfadd.vv v16, v0, v8");

        asm volatile ("vse16.v v16, (%0)" :: "r"(r));

        a += vl;
        b += vl;
        r += vl;
    }

    return 0;
}
