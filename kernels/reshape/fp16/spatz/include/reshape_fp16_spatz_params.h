#ifndef RESHAPE_FP16_SPATZ_PARAMS_H_
#define RESHAPE_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X;      /* Tile's input tensor shard        */
    uintptr_t shard_Y;      /* Tile's output tensor shard       */

    uint32_t start_idx;     /* Start element index for this tile*/
    uint32_t len;           /* Number of elements for this tile */
} reshape_fp16_spatz_params_t;

#endif  /* RESHAPE_FP16_SPATZ_PARAMS_H_ */
