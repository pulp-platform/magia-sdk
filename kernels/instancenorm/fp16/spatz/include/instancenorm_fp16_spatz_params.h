#ifndef INSTANCENORM_FP16_SPATZ_PARAMS_H_
#define INSTANCENORM_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_input;      /* Tile's input tensor shard                */
    uintptr_t shard_output;     /* Tile's output tensor shard               */

    uintptr_t gamma;            /* InstanceNorm scale tensor                */
    uintptr_t beta;             /* InstanceNorm bias (B) tensor             */
    uintptr_t eps;              /* InstanceNorm epsilon scalar              */

    uint32_t inst_start;        /* Tile's start instance index (global)    */
    uint32_t inst_len;          /* Number of instances for this tile       */
    uint32_t hw_len;            /* Elements per instance (H * W)           */
    uint32_t num_channels;      /* Total number of channels (C)            */
} instancenorm_fp16_spatz_params_t;

#endif  /* INSTANCENORM_FP16_SPATZ_PARAMS_H_ */
