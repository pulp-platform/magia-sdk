#ifndef LAYERNORM_FP16_SPATZ_PARAMS_H_
#define LAYERNORM_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X;      /* Tile's input tensor shard            */
    uintptr_t shard_Y;      /* Tile's output tensor shard           */

    uintptr_t gamma;        /* LayerNorm scale tensor (Dim: W)      */
    uintptr_t beta;         /* LayerNorm bias (B) tensor (Dim: W)   */
    uintptr_t eps;          /* LayerNorm epsilon scalar             */

    uint32_t r_start;       /* Tile's row shard start (global)      */
    uint32_t r_len;         /* Tile's number of rows to process     */
    uint32_t w_len;         /* Elements per row (W dimension)       */
} layernorm_fp16_spatz_params_t;

#endif  /* LAYERNORM_FP16_SPATZ_PARAMS_H_ */
