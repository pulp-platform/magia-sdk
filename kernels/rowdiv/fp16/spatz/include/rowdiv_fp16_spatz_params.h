#ifndef ROWDIV_FP16_SPATZ_PARAMS_H_
#define ROWDIV_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X; /* Tile's input rows shard              */
    uintptr_t shard_S; /* Tile's per-row divisors shard        */
    uintptr_t shard_Y; /* Tile's output rows shard             */

    uint32_t cols;      /* Row length                          */
    uint32_t row_start; /* Tile's row shard start              */
    uint32_t row_len;   /* Tile's number of rows               */
} rowdiv_fp16_spatz_params_t;

#endif /* ROWDIV_FP16_SPATZ_PARAMS_H_ */
