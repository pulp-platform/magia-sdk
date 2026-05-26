#ifndef SOFTMAX_FP16_SPATZ_PARAMS_H_
#define SOFTMAX_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_input;  /* Tile's input tensor shard            */
    uintptr_t shard_output; /* Tile's output tensor shard           */

    uint32_t r_start;       /* Tile's row shard start (global)      */
    uint32_t r_len;         /* Tile's number of rows to process     */
    uint32_t w_len;         /* Elements per row (W dimension)       */
} softmax_fp16_spatz_params_t;

#endif  /* SOFTMAX_FP16_SPATZ_PARAMS_H_ */
