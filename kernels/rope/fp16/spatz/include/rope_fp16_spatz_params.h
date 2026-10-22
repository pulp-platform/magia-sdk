#ifndef ROPE_FP16_SPATZ_PARAMS_H_
#define ROPE_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X; /* Tile's input rows [r_len x n_heads * head_dim]  */
    uintptr_t shard_Y; /* Tile's output rows (may alias shard_X)          */
    uintptr_t cos;     /* Tile's cos rows [r_len x head_dim / 2]          */
    uintptr_t sin;     /* Tile's sin rows [r_len x head_dim / 2]          */

    uint32_t r_start;  /* Tile's row (token) shard start  */
    uint32_t r_len;    /* Tile's number of rows (tokens)  */
    uint32_t n_heads;  /* Heads per row                   */
    uint32_t head_dim; /* Head size (even)                */
} rope_fp16_spatz_params_t;

#endif /* ROPE_FP16_SPATZ_PARAMS_H_ */
