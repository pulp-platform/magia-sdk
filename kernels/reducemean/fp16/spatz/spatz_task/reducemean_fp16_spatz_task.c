#include "tile.h"
#include "reducemean_fp16_spatz_params.h"

static inline void reduce_mean_core(const _Float16 *src, _Float16 *dst, const size_t reduce_dim, const size_t inner_dim, const _Float16 scale_factor)
{
    size_t avl = inner_dim;
    size_t vl;

    for (; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        asm volatile ("vmv.v.i v0, 0");

        for (size_t r = 0; r < reduce_dim; r++) {
            const _Float16 *p_src = src + (r * inner_dim) + (inner_dim - avl);

            asm volatile ("vle16.v v8, (%0)" :: "r"(p_src));
            asm volatile ("vfadd.vv v0, v0, v8");
        }

        asm volatile ("vfmul.vf v0, v0, %0" :: "f"(scale_factor));

        _Float16 *p_dst = dst + (inner_dim - avl);
        asm volatile ("vse16.v v0, (%0)" :: "r"(p_dst) : "memory");
    }
}

int reducemean_fp16_spatz_task(void)
{
    volatile reducemean_fp16_spatz_params_t *params;
    uintptr_t params_addr;

    const _Float16 *src;
    _Float16 *dst;
    uint32_t reduce_dim;
    uint32_t inner_dim;
    uint32_t outer_len;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile reducemean_fp16_spatz_params_t *) params_addr;

    src = (_Float16 *) params->shard_X;
    dst = (_Float16 *) params->shard_Y;
    reduce_dim = params->reduce_dim;
    inner_dim = params->inner_dim;
    outer_len = params->outer_len;

    _Float16 scale_factor = (_Float16)1.0 / (_Float16)reduce_dim;

    uint32_t in_stride  = reduce_dim * inner_dim;
    uint32_t out_stride = inner_dim;

    for (uint32_t o = 0; o < outer_len; o++) {
        reduce_mean_core(src + (o * in_stride), dst + (o * out_stride), reduce_dim, inner_dim, scale_factor);
    }

    return 0;
}
