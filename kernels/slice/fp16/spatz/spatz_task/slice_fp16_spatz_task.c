#include "tile.h"
#include "slice_fp16_spatz_params.h"

static inline void slice(const _Float16 *src, _Float16 *dst, const size_t len)
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

int slice_fp16_spatz_task(void)
{
    volatile slice_fp16_spatz_params_t *params;
    uintptr_t params_addr;

    const _Float16 *src_base;
    _Float16 *dst_base;

    uint32_t slice_dim;
    uint32_t out_slice_dim;
    uint32_t inner_dim;
    uint32_t start_idx;
    uint32_t len_outer;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile slice_fp16_spatz_params_t *) params_addr;

    src_base = (_Float16 *) params->shard_X;
    dst_base = (_Float16 *) params->shard_Y;
    slice_dim = params->slice_dim;
    out_slice_dim = params->out_slice_dim;
    inner_dim = params->inner_dim;
    start_idx = params->start_idx;
    len_outer = params->len_outer;

    for (uint32_t o = 0; o < len_outer; o++) {
        const _Float16 *current_src;
        _Float16 *current_dst;

        current_src = src_base + (o * slice_dim * inner_dim) + (start_idx * inner_dim);
        current_dst = dst_base + (o * out_slice_dim * inner_dim);

        slice(current_src, current_dst, out_slice_dim * inner_dim);
    }

    return 0;
}
