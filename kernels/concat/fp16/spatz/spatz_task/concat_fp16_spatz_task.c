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

    const _Float16 *src_input0;
    const _Float16 *src_input1;
    _Float16 *dst_result;

    uint32_t iter_len;
    uint32_t len_input0;
    uint32_t len_input1;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile concat_fp16_spatz_params_t *) params_addr;

    src_input0 = (_Float16 *) params->shard_input0;
    src_input1 = (_Float16 *) params->shard_input1;
    dst_result = (_Float16 *) params->shard_output;
    len_input0 = params->len_input0;
    len_input1 = params->len_input1;
    iter_len   = params->iter_len;

    for (int i = 0; i < iter_len; i++) {
        concat(src_input0, dst_result, len_input0);
        src_input0 += len_input0;
        dst_result += len_input0;

        concat(src_input1, dst_result, len_input1);
        src_input1 += len_input1;
        dst_result += len_input1;
    }

    return 0;
}
