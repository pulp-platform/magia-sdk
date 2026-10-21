#include "tile.h"
#include "memset_fp16_spatz_params.h"

int memset_fp16_spatz_task(void)
{
    volatile memset_fp16_spatz_params_t *params;
    uintptr_t params_addr;
    _Float16 *C;
    size_t avl;
    size_t vl;

    params_addr = mmio32(SPATZ_DATA);
    params      = (volatile memset_fp16_spatz_params_t *)params_addr;

    C   = (_Float16 *)params->shard_C;
    avl = params->len;

    asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vmv.v.i v0, 0");

    for (; avl > 0; avl -= vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        asm volatile("vse16.v v0, (%0)" ::"r"(C) : "memory");

        C += vl;
    }

    return 0;
}
