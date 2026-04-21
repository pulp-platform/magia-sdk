#include "tile.h"
#include "onnx_div_params.h"

int onnx_div_task(void)
{
    volatile onnx_div_params_t *params;
    uintptr_t params_addr;
    _Float16 *A;
    _Float16 *B;
    _Float16 *C;
    size_t avl;
    size_t vl;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile onnx_div_params_t *) params_addr;

    A = (_Float16 *)params->chunk_A;
    B = (_Float16 *)params->chunk_B;
    C = (_Float16 *)params->chunk_C;
    avl = params->len;

    for (; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        asm volatile ("vle16.v v0, (%0)" :: "r"(A));
        asm volatile ("vle16.v v8, (%0)" :: "r"(B));

        asm volatile ("vfdiv.vv v16, v0, v8");

        asm volatile ("vse16.v v16, (%0)" :: "r"(C));

        A += vl;
        B += vl;
        C += vl;
    }

    return 0;
}
