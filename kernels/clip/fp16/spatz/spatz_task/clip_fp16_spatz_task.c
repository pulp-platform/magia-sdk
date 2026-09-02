#include "tile.h"
#include "clip_fp16_spatz_params.h"

int clip_fp16_spatz_task(void)
{
    volatile clip_fp16_spatz_params_t *params;
    uintptr_t params_addr;
    _Float16 *output;
    _Float16 *input;
    _Float16 min;
    _Float16 max;
    size_t avl;
    size_t vl;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile clip_fp16_spatz_params_t *) params_addr;

    input = (_Float16 *)params->shard_input;
    output = (_Float16 *)params->shard_output;
    min = *(_Float16 *)params->min;
    max = *(_Float16 *)params->max;
    avl = params->len;

    for (; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        asm volatile ("vle16.v v0, (%0)" :: "r"(input));

        asm volatile ("vfmin.vf v8, v0, %0" :: "f"(max));
        asm volatile ("vfmax.vf v8, v8, %0" :: "f"(min));

        asm volatile ("vse16.v v8, (%0)" :: "r"(output) : "memory");

        output += vl;
        input += vl;
    }

    return 0;
}
