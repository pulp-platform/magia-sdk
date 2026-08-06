#include "tile.h"
#include "concat_fp16_spatz_params.h"

static inline void concat(const _Float16 *src, _Float16 *dst, const size_t len)
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

int concat_fp16_spatz_task(void)
{
    volatile concat_fp16_spatz_params_t *params;
    uintptr_t params_addr;

    const uintptr_t *shard_input;
    const uint32_t *len_input;
    _Float16 *dst_result;

    uint32_t iter_len;
    uint32_t num_inputs;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile concat_fp16_spatz_params_t *) params_addr;

    shard_input = (const uintptr_t *) params->shard_input;
    len_input   = (const uint32_t *) params->len_input;
    dst_result  = (_Float16 *) params->shard_output;
    iter_len    = params->iter_len;
    num_inputs  = params->num_inputs;

    for (int i = 0; i < iter_len; i++) {
        for (int k = 0; k < num_inputs; k++) {
            const _Float16 *src = (const _Float16 *) shard_input[k] + i * len_input[k];
            concat(src, dst_result, len_input[k]);
            dst_result += len_input[k];
        }
    }

    return 0;
}
