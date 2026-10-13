#ifndef GROUP_NORMALIZE_FP16_SPATZ_PARAMS_H_
#define GROUP_NORMALIZE_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t input;
    uintptr_t mean;
    uintptr_t variance;
    uintptr_t scale;
    uintptr_t bias;
    uintptr_t output;
    uint32_t local_spatial_elements;
    uint32_t local_elements;
    uint32_t channel_offset;
    uint32_t scale_channel_offset;
    uint32_t num_groups;
    uint32_t channels_per_group;
    float epsilon;
} group_normalize_fp16_spatz_params_t;

#endif
