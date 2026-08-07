#ifndef SOFTMAX_FP16_SPATZ_PARAMS_H_
#define SOFTMAX_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_input;  /* Tile's input tensor shard            */
    uintptr_t shard_output; /* Tile's output tensor shard           */

    uint32_t reduce_dim;    /* Elements to normalize over (axis)    */
    uint32_t inner_dim;     /* Inner (post-axis) dimension block    */
    uint32_t outer_start;   /* Tile's outer loop start index        */
    uint32_t outer_len;     /* Number of outer loops for tile       */
} softmax_fp16_spatz_params_t;

#endif  /* SOFTMAX_FP16_SPATZ_PARAMS_H_ */
