#include "tile.h"
#include "onnx_floor_params.h"

// Rounding Mode: Round Down
#define RDN 0x2

int onnx_floor_task(void)
{
    volatile onnx_floor_params_t *params;
    uintptr_t params_addr;
    _Float16 *src;
    _Float16 *dst;
    size_t avl;
    size_t vl;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile onnx_floor_params_t *) params_addr;

    src = (_Float16 *)params->addr_input;
    dst = (_Float16 *)params->addr_res;
    avl = params->len;

    asm volatile ("fsrm %0" ::"r"(RDN));

    for (; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile ("vle16.v v8, (%0)" :: "r"(src));
        asm volatile ("vfcvt.x.f.v v16, v8");
        asm volatile ("vfcvt.f.x.v v8, v16");
        asm volatile ("vse16.v v8, (%0)" :: "r"(dst) : "memory");

        src += vl;
        dst += vl;
    }

    return 0;
}
