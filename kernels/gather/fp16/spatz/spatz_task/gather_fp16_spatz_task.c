#include "tile.h"
#include "gather_fp16_spatz_params.h"

static inline void gather(const _Float16 *src, _Float16 *dst, const size_t len)
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

int gather_fp16_spatz_task(void)
{
    volatile gather_fp16_spatz_params_t *params;
    uintptr_t params_addr;

    const _Float16 *input;
    _Float16 *output;

    uint32_t batch_len;
    uint32_t gather_dim_size;
    uint32_t axis_length;
    uint32_t index;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile gather_fp16_spatz_params_t *) params_addr;

    input = (_Float16 *) params->shard_input;
    output = (_Float16 *) params->shard_output;
    gather_dim_size = params->gather_dim_size;
    axis_length = params->axis_length;
    batch_len = params->batch_len;
    index = params->index;

    uint32_t in_batch_stride = gather_dim_size * axis_length;
    uint32_t target_fetta_offset = index * axis_length;

    for (int b = 0; b < batch_len; b++) {
        const _Float16 *fetta_ptr = input + target_fetta_offset;

        gather(fetta_ptr, output, axis_length);

        input += in_batch_stride;
        output += axis_length;
    }

    return 0;
}
