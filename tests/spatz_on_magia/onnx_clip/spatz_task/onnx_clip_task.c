#include "onnx_clip_params.h"

#include "magia_spatz_utils.h"
#include "magia_tile_utils.h"

int onnx_clip_task(void)
{
    volatile onnx_clip_params_t *params;
    uintptr_t params_addr;

    _Float16 *result;
    _Float16 *input;
    _Float16 min;
    _Float16 max;

    size_t avl;
    size_t vl;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile onnx_clip_params_t *) params_addr;

    input = (_Float16 *)params->addr_input;
    result = (_Float16 *)params->addr_res;
    min = *(_Float16 *)params->addr_min;
    max = *(_Float16 *)params->addr_max;
    avl = params->len;

    for (; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        asm volatile ("vle16.v v0, (%0)" :: "r"(input));

        asm volatile ("vfmin.vf v8, v0, %0" :: "f"(max));
        asm volatile ("vfmax.vf v8, v8, %0" :: "f"(min));

        asm volatile ("vse16.v v8, (%0)" :: "r"(result));

        result += vl;
        input += vl;
    }

    return 0;
}
