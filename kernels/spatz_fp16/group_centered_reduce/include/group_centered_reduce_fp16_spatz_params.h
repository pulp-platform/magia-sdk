#ifndef GROUP_CENTERED_REDUCE_FP16_SPATZ_PARAMS_H_
#define GROUP_CENTERED_REDUCE_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t input;
    uintptr_t mean;
    uintptr_t output;
    uint32_t local_elements;
    uint32_t local_spatial_elements;
    uint32_t channel_offset;
    uint32_t elements_per_group;
    uint32_t channels_per_group;
    uint32_t num_groups;
} group_centered_reduce_fp16_spatz_params_t;

#endif
