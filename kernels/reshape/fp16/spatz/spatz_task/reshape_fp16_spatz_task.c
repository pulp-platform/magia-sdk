#include "tile.h"
#include "reshape_fp16_spatz_params.h"

static inline void reshape(const _Float16 *src, _Float16 *dst, const size_t len)
{
    const _Float16 *p_src;
    _Float16 *p_dst;
    size_t avl;
    size_t vl;

    p_src = src;
    p_dst = dst;
    avl = len;

    for (; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        asm volatile ("vle16.v v0, (%0)" :: "r"(p_src));

        asm volatile ("vse16.v v0, (%0)" :: "r"(p_dst) : "memory");

        p_src += vl;
        p_dst += vl;
    }
}

int reshape_fp16_spatz_task(void)
{
    volatile reshape_fp16_spatz_params_t *params;
    uintptr_t params_addr;

    const _Float16 *src;
    _Float16 *dst;

    uint32_t len;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile reshape_fp16_spatz_params_t *) params_addr;

    src = (_Float16 *) params->shard_X;
    dst = (_Float16 *) params->shard_Y;
    len = params->len;

    reshape(src, dst, len);

    return 0;
}
