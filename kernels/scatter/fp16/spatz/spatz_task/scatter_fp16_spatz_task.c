#include "tile.h"
#include "scatter_fp16_spatz_params.h"

int scatter_fp16_spatz_task(void)
{
    volatile scatter_fp16_spatz_params_t *params;
    uintptr_t params_addr;

    _Float16 *p_updates;
    _Float16 *p_data;
    _Float16 *p_out;
    int32_t *p_idx;

    uint32_t elems_per_tile;
    uint32_t outer_per_tile;
    uint32_t iter_start;
    uint32_t inner_size;
    uint32_t data_axis_dim;
    uint32_t indices_axis_dim;
    size_t avl;
    size_t vl;


    params_addr = mmio32(SPATZ_DATA);
    params = (volatile scatter_fp16_spatz_params_t *) params_addr;

    p_updates = (_Float16 *) params->shard_updates;
    p_out = (_Float16 *) params->shard_output;
    p_data = (_Float16 *) params->shard_data;
    p_idx = (int32_t *)  params->shard_indices;

    elems_per_tile = params->elems_per_tile;
    outer_per_tile = params->outer_per_tile;
    iter_start = params->outer_start;
    inner_size = params->inner_size;
    data_axis_dim = params->data_axis_dim;
    indices_axis_dim = params->indices_axis_dim;

    /* Step 1: Copy */
    avl = elems_per_tile;
    _Float16 *src_ptr = p_data;
    _Float16 *dst_ptr = p_out;
    while (avl > 0) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile ("vle16.v v0, (%0)" :: "r"(src_ptr));
        asm volatile ("vse16.v v0, (%0)" :: "r"(dst_ptr) : "memory");
        src_ptr += vl;
        dst_ptr += vl;
        avl -= vl;
    }


    /* Step 2: Scatter (RVV in this case makes no sense) */
    for (uint32_t out_idx = 0; out_idx < outer_per_tile; out_idx++) {
        for (uint32_t ax_idx = 0; ax_idx < indices_axis_dim; ax_idx++) {
            for (uint32_t in_idx = 0; in_idx < inner_size; in_idx++) {

                uint32_t current_offset = (out_idx * indices_axis_dim + ax_idx) * inner_size + in_idx;
                int32_t target_axis_idx = p_idx[current_offset];

                if (target_axis_idx < 0) {
                    target_axis_idx += (int32_t)data_axis_dim;
                }

                uint32_t target_offset = (out_idx * data_axis_dim + (uint32_t)target_axis_idx) * inner_size + in_idx;
                p_out[target_offset] = p_updates[current_offset];
            }
        }
    }

    return 0;
}
