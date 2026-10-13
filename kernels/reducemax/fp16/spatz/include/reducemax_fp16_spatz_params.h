#ifndef REDUCEMAX_FP16_SPATZ_PARAMS_H_
#define REDUCEMAX_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X;
    uintptr_t shard_Y;
    uint32_t reduce_dim;
    uint32_t inner_dim;
} reducemax_fp16_spatz_params_t;

#endif
