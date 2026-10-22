#include "tile.h"
#include "utils/spatz_recip.h"
#include "div_fp16_spatz_params.h"

int div_fp16_spatz_task(void)
{
    volatile div_fp16_spatz_params_t *params;
    uintptr_t params_addr;
    _Float16 *A;
    _Float16 *B;
    _Float16 *C;
    size_t avl;
    size_t vl;

    params_addr = mmio32(SPATZ_DATA);
    params      = (volatile div_fp16_spatz_params_t *)params_addr;

    A   = (_Float16 *)params->shard_A;
    B   = (_Float16 *)params->shard_B;
    C   = (_Float16 *)params->shard_C;
    avl = params->len;

    for (; avl > 0; avl -= vl) {
        asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(avl)); /* m4: room for the division */

        asm volatile("vle16.v v0, (%0)" ::"r"(A));
        asm volatile("vle16.v v8, (%0)" ::"r"(B));

        SPATZ_VFDIV_VV("v16", "v0", "v8", "v20", "v24"); /* no vfdiv, see spatz_recip.h */

        asm volatile("vse16.v v16, (%0)" SPATZ_VST_FENCE ::"r"(C));

        A += vl;
        B += vl;
        C += vl;
    }

    return 0;
}
