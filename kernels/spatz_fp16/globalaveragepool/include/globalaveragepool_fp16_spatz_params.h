#ifndef GLOBALAVERAGEPOOL_FP16_SPATZ_PARAMS_H_
#define GLOBALAVERAGEPOOL_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X;      /* Tile's input tensor shard (L1)       */
    uintptr_t shard_Y;      /* Tile's output tensor shard (L1)      */

    uint32_t start;         /* Tile's start global index            */
    uint32_t len;           /* Number of elements for this tile     */
    uint32_t hw_len;        /* Elements per shard                   */
} globalaveragepool_fp16_spatz_params_t;

#endif  /* GLOBALAVERAGEPOOL_FP16_SPATZ_PARAMS_H_ */
