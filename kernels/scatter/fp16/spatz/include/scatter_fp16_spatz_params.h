#ifndef SCATTER_FP16_SPATZ_PARAMS_H_
#define SCATTER_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_data;
    uintptr_t shard_indices;
    uintptr_t shard_updates;
    uintptr_t shard_output;

    uint32_t outer_per_tile;
    uint32_t elems_per_tile;
    uint32_t elems_indices_per_tile;
    uint32_t outer_start;
    uint32_t inner_size;
    uint32_t data_axis_dim;
    uint32_t indices_axis_dim;
    uint32_t axis;
} scatter_fp16_spatz_params_t;

#endif  /* SCATTER_FP16_SPATZ_PARAMS_H_ */
