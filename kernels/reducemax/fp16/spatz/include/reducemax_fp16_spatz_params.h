#ifndef REDUCEMAX_FP16_SPATZ_PARAMS_H_
#define REDUCEMAX_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X; /* Tile's input data shard         */
    uintptr_t shard_Y; /* Tile's output data shard        */

    uint32_t reduce_dim;  /* Elements to collapse (max over) */
    uint32_t inner_dim;   /* Spatial/inner dimension block   */
    uint32_t outer_start; /* Tile's outer loop start index   */
    uint32_t outer_len;   /* Number of outer loops for tile  */
} reducemax_fp16_spatz_params_t;

#endif /* REDUCEMAX_FP16_SPATZ_PARAMS_H_ */
