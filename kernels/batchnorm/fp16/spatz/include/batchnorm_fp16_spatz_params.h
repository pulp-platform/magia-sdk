#ifndef BATCHNORM_FP16_SPATZ_PARAMS_H_
#define BATCHNORM_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X;      /* Tile's input tensor shard        */
    uintptr_t shard_Y;      /* Tile's output tensor shard       */

    uintptr_t gamma;        /* BatchNorm scale tensor           */
    uintptr_t beta;         /* BatchNorm bias (B) tensor        */
    uintptr_t mean;         /* BatchNorm mean tensor            */
    uintptr_t var;          /* BatchNorm variance tensor        */
    uintptr_t eps;          /* BatchNorm epsilon scalar         */

    uint32_t channels;      /* Total number of input channels   */
    uint32_t c_start;       /* Tile's channel shard start       */
    uint32_t c_len;         /* Tile's number of channels        */
    uint32_t hw_len;        /* Elements per channel             */
} batchnorm_fp16_spatz_params_t;

#endif  /* BATCHNORM_FP16_SPATZ_PARAMS_H_ */
