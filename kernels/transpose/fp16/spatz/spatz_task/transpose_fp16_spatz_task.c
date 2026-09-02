#include "tile.h"
#include "transpose_fp16_spatz_params.h"

static inline void transpose(const _Float16 *src, _Float16 *dst, const size_t len, const size_t stride)
{
    const _Float16 *p_src = src;
    _Float16 *p_dst = dst;
    size_t avl = len;
    size_t vl;

    size_t stride_bytes = stride * sizeof(_Float16);

    for (; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile ("vlse16.v v0, (%0), %1" :: "r"(p_src), "r"(stride_bytes));
        asm volatile ("vse16.v v0, (%0)" :: "r"(p_dst) : "memory");

        p_src += vl * stride;
        p_dst += vl;
    }
}

#if 0
static inline void transpose_scalar(const _Float16 *src, _Float16 *dst, const size_t len, const size_t stride)
{
    for (size_t i = 0; i < len; i++)
        dst[i] = src[i * stride];
}
#endif

int transpose_fp16_spatz_task(void)
{
    volatile transpose_fp16_spatz_params_t *params;
    uintptr_t params_addr;


    const _Float16 *shard_input;
    _Float16 *shard_output;
    uint32_t *out_shape;
    uint32_t *in_strides;
    uint32_t *perm;
    uint32_t *coord;
    uint32_t iter_len;
    uint32_t rank;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile transpose_fp16_spatz_params_t *) params_addr;

    shard_input  = (_Float16 *) params->shard_input;
    shard_output = (_Float16 *) params->shard_output;
    out_shape    = (uint32_t *) params->out_shape;
    in_strides   = (uint32_t *) params->in_strides;
    perm         = (uint32_t *) params->perm;
    coord        = (uint32_t *) params->coord;
    iter_len     = params->iteration_len;
    rank         = params->rank;

    uint32_t inner_len    = out_shape[rank - 1];
    uint32_t inner_stride = in_strides[perm[rank - 1]];

    uint32_t middle_count = 1;
    for (uint32_t k = 1; k + 1 < rank; k++)
        middle_count *= out_shape[k];

    for (uint32_t i0 = 0; i0 < iter_len; i0++) {
        for (uint32_t k = 1; k < rank; k++)
            coord[k] = 0;
        coord[0] = i0;

        for (uint32_t m = 0; m < middle_count; m++) {
            uint32_t base_in_offset = 0;
            for (uint32_t k = 0; k < rank; k++)
                base_in_offset += coord[k] * in_strides[perm[k]];

#if 0
            /* Spatz VLSU corrupts vector stores to non-aligned addresses */
            if (((uintptr_t) shard_output & 3u) == 0)
                transpose(shard_input + base_in_offset, shard_output, inner_len, inner_stride);
            else
                transpose_scalar(shard_input + base_in_offset, shard_output, inner_len, inner_stride);
#else
            transpose(shard_input + base_in_offset, shard_output, inner_len, inner_stride);
#endif

            shard_output += inner_len;

            for (int k = (int) rank - 2; k >= 1; k--) {
                if (++coord[k] < out_shape[k])
                    break;
                coord[k] = 0;
            }
        }
    }

    return 0;
}
