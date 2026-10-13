#include "tile.h"
#include "ceil_fp16_spatz_params.h"

// Rounding Mode: Round UP
#define RUP 0x3

int ceil_fp16_spatz_task(void)
{
    volatile ceil_fp16_spatz_params_t *params;
    uintptr_t params_addr;
    _Float16 *X;
    _Float16 *Y;
    size_t avl;
    size_t vl;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile ceil_fp16_spatz_params_t *) params_addr;

    X = (_Float16 *)params->shard_X;
    Y = (_Float16 *)params->shard_Y;
    avl = params->len;

    asm volatile ("fsrm %0" ::"r"(RUP));

    for (; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile ("vle16.v v8, (%0)" :: "r"(X));
        asm volatile ("vfcvt.x.f.v v16, v8");
        asm volatile ("vfcvt.f.x.v v8, v16");
        asm volatile ("vse16.v v8, (%0)" :: "r"(Y) : "memory");

        X += vl;
        Y += vl;
    }

    return 0;
}
