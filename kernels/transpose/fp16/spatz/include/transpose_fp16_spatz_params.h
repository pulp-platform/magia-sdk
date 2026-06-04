#ifndef TRANSPOSE_FP16_SPATZ_PARAMS_H_
#define TRANSPOSE_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_input;          /* Tile's input data tensor shard   */
    uintptr_t shard_output;         /* Tile's output tensor shard       */

    uint32_t out_shape[4];          /* 4D Output tensor shape           */
    uint32_t in_strides[4];         /* 4D Input tensor original strides */
    uint32_t perm[4];               /* Axis permutation vector          */

    uint32_t iteration_start;       /* Tile's loop i0 shard start index */
    uint32_t iteration_len;         /* Tile's number of iterations      */
} transpose_fp16_spatz_params_t;

#endif /* TRANSPOSE_FP16_SPATZ_PARAMS_H_ */
