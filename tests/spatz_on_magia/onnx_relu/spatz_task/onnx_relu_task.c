#include "tile.h"
#include "onnx_relu_params.h"

int onnx_relu_task(void)
{
    register _Float16 ZERO asm ("fs0") = 0.0f;
    volatile onnx_relu_params_t *params;
    uintptr_t params_addr;
    _Float16 *X;
    _Float16 *Y;
    size_t avl;
    size_t vl;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile onnx_relu_params_t *) params_addr;

    X = (_Float16 *)params->chunk_X;
    Y = (_Float16 *)params->chunk_Y;
    avl = params->len;

    for (; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile ("vle16.v v0, (%0)" :: "r"(X));
        asm volatile ("vfmax.vf v8, v0, %0" :: "f"(ZERO));
        asm volatile ("vse16.v v8, (%0)" :: "r"(Y));

        X += vl;
        Y += vl;
    }

    return 0;
}
