#ifndef BIAS_ROPE_T_FP16_SPATZ_PARAMS_H_
#define BIAS_ROPE_T_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X; /* Transposed block [rows x cols], cols = tokens, updated in place */
    uintptr_t bias;    /* Bias of the block's rows [rows]                                */
    uintptr_t cos;     /* cos of the block's tokens, transposed [head_dim / 2 x cols]     */
    uintptr_t sin;     /* sin of the block's tokens, transposed [head_dim / 2 x cols]     */

    uint32_t dims;     /* Number of rows (even) | number of columns << 16 (even) */
    uint32_t row0;     /* Row of the block's first row in its head (multiple of group) */
    uint32_t head_dim; /* Head size (even) */
    uint32_t group;    /* RoPE group G: pairs (c, c + G/2) inside groups of G channels */
} bias_rope_t_fp16_spatz_params_t;

#endif /* BIAS_ROPE_T_FP16_SPATZ_PARAMS_H_ */
