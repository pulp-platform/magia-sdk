#ifndef LAYERNORM_FP16_SPATZ_PARAMS_H_
#define LAYERNORM_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X;       /* Tile's input tensor shard                     */
    uintptr_t shard_Y;       /* Tile's output tensor shard                    */

    uintptr_t gamma;         /* LayerNorm scale tensor                        */
    uintptr_t beta;          /* LayerNorm bias tensor                         */
    uintptr_t eps;           /* LayerNorm epsilon scalar                      */

    uint32_t r_start;        /* Tile's row shard start                        */
    uint32_t r_len;          /* Tile's number of rows                         */
    uint32_t w_len;          /* Row length (size of normalized axis)          */

    uint32_t input_len;      /* Total elements of the input tensor            */
    uint32_t output_len;     /* Local elements of the output shard            */
} layernorm_fp16_spatz_params_t;

#endif  /* LAYERNORM_FP16_SPATZ_PARAMS_H_ */
