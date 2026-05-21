#ifndef GROUPNORM_FP16_SPATZ_PARAMS_H_
#define GROUPNORM_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X;      /* Tile's input tensor shard        */
    uintptr_t shard_Y;      /* Tile's output tensor shard       */

    uintptr_t gamma;        /* GroupNorm scale tensor           */
    uintptr_t beta;         /* GroupNorm bias tensor (all C)    */
    uintptr_t eps;          /* GroupNorm epsilon scalar         */

    uint32_t g_start;       /* Tile's group shard start         */
    uint32_t g_len;         /* Tile's number of groups          */
    uint32_t c_per_g;       /* Number of channels per group     */
    uint32_t hw_len;        /* Elements per channel (H * W)     */
} groupnorm_fp16_spatz_params_t;

#endif  /* GROUPNORM_FP16_SPATZ_PARAMS_H_ */
