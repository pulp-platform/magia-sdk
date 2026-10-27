#ifndef GELU_FAST_FP16_SPATZ_PARAMS_H_
#define GELU_FAST_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X; /* Input [len], overwritten with the output if shard_Y == shard_X */
    uintptr_t shard_Y; /* Output [len] (may alias X)                                    */

    uint32_t len;      /* Number of elements */
} gelu_fast_fp16_spatz_params_t;

#endif /* GELU_FAST_FP16_SPATZ_PARAMS_H_ */
