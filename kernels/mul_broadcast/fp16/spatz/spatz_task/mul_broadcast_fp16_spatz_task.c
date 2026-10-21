#include "tile.h"
#include "mul_broadcast_fp16_spatz_params.h"

int mul_broadcast_fp16_spatz_task(void)
{
    volatile mul_broadcast_fp16_spatz_params_t *params;
    uintptr_t params_addr;
    const _Float16 *A;
    _Float16 *C;
    _Float16 s;
    size_t avl;
    size_t vl;

    params_addr = mmio32(SPATZ_DATA);
    params      = (volatile mul_broadcast_fp16_spatz_params_t *)params_addr;
    A           = (_Float16 *)params->shard_A;
    C           = (_Float16 *)params->shard_C;
    s           = *(_Float16 *)params->scalar;
    avl         = params->elem_len;

    for (; avl > 0; avl -= vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        asm volatile("vle16.v v0, (%0)" ::"r"(A));

        asm volatile("vfmul.vf v0, v0, %0" ::"f"(s));

        asm volatile("vse16.v v0, (%0)" ::"r"(C) : "memory");

        A += vl;
        C += vl;
    }

    return 0;
}
