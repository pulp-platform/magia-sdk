#ifndef BIAS_ROPE_FP16_SPATZ_PARAMS_H_
#define BIAS_ROPE_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X; /* Block [rows x cols], rows = tokens, updated in place           */
    uintptr_t bias;    /* Bias of the block's columns [cols]                             */
    uintptr_t cos;     /* cos of the block's tokens [rows x head_dim / 2], 0: bias only   */
    uintptr_t sin;     /* sin of the block's tokens [rows x head_dim / 2]                */

    uint32_t dims;     /* Number of rows | number of columns << 16 (even) */
    uint32_t col0;     /* Column of the block's first column in its head (multiple of group) */
    uint32_t head_dim; /* Head size (even) */
    uint32_t group;    /* RoPE group G: pairs (c, c + G/2) inside groups of G channels */
} bias_rope_fp16_spatz_params_t;

#endif /* BIAS_ROPE_FP16_SPATZ_PARAMS_H_ */
