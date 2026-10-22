#ifndef ROWSCALE_FP16_SPATZ_PARAMS_H_
#define ROWSCALE_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X; /* Tile's input rows shard              */
    uintptr_t shard_S; /* Tile's per-row scale factors shard   */
    uintptr_t shard_Y; /* Tile's output rows shard             */

    uint32_t cols;      /* Row length                          */
    uint32_t row_start; /* Tile's row shard start              */
    uint32_t row_len;   /* Tile's number of rows               */
} rowscale_fp16_spatz_params_t;

#endif /* ROWSCALE_FP16_SPATZ_PARAMS_H_ */
