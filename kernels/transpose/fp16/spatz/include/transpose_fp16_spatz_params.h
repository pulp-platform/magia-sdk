#ifndef TRANSPOSE_FP16_SPATZ_PARAMS_H_
#define TRANSPOSE_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_input;          /* Tile's input data tensor shard   */
    uintptr_t shard_output;         /* Tile's output tensor shard       */

    uintptr_t out_shape;            /* uint32_t[rank] output shape      */
    uintptr_t in_strides;           /* uint32_t[rank] input strides     */
    uintptr_t perm;                 /* uint32_t[rank] axis permutation  */
    uintptr_t coord;                /* uint32_t[rank] odometer scratch  */

    uint32_t rank;                  /* Tensor rank                      */
    uint32_t iteration_start;       /* Tile's loop i0 shard start index */
    uint32_t iteration_len;         /* Tile's number of iterations      */
} transpose_fp16_spatz_params_t;

#endif /* TRANSPOSE_FP16_SPATZ_PARAMS_H_ */
