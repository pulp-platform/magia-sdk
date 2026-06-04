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

int transpose_fp16_spatz_task(void)
{
    volatile transpose_fp16_spatz_params_t *params;
    uintptr_t params_addr;
    const _Float16 *shard_input;
    _Float16 *shard_output;
    uint32_t in_strides[4];
    uint32_t out_shape[4];
    uint32_t iter_start;
    uint32_t iter_len;
    uint32_t perm[4];

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile transpose_fp16_spatz_params_t *) params_addr;

    shard_input  = (_Float16 *) params->shard_input;
    shard_output = (_Float16 *) params->shard_output;
    iter_start   = params->iteration_start;
    iter_len     = params->iteration_len;

    for (int i = 0; i < 4; i++) {
        out_shape[i]  = params->out_shape[i];
        in_strides[i] = params->in_strides[i];
        perm[i]       = params->perm[i];
    }

    uint32_t coord_in[4] = {0, 0, 0, 0};

    for (uint32_t i0 = 0; i0 < iter_len; i0++) {
        for (uint32_t i1 = 0; i1 < out_shape[1]; i1++) {
            for (uint32_t i2 = 0; i2 < out_shape[2]; i2++) {
                coord_in[perm[0]] = i0;
                coord_in[perm[1]] = i1;
                coord_in[perm[2]] = i2;
                coord_in[perm[3]] = 0;

                uint32_t base_in_offset = 0;
                for (int j = 0; j < 4; j++) {
                    base_in_offset += coord_in[j] * in_strides[j];
                }

                transpose(shard_input + base_in_offset, shard_output, out_shape[3], in_strides[perm[3]]);

                shard_output += out_shape[3];
            }
        }
    }

    return 0;
}
