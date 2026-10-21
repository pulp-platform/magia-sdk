#ifndef MEMSET_FP16_SPATZ_PARAMS_H_
#define MEMSET_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_C; /* Tile's output tensor C shard */

    uint32_t len; /* Number of elements */
} memset_fp16_spatz_params_t;

#endif /* MEMSET_FP16_SPATZ_PARAMS_H_ */
