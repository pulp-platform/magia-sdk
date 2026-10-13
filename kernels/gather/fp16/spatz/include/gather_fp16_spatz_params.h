#ifndef GATHER_FP16_SPATZ_PARAMS_H_
#define GATHER_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_input;          /* Tile's input data tensor shard  */
    uintptr_t shard_output;         /* Tile's output tensor shard      */

    uint32_t batch_start;           /* Tile's batch shard start index  */
    uint32_t batch_len;             /* Tile's number of batches        */
    uint32_t gather_dim_size;       /* Size of the gather axis         */
    uint32_t axis_length;           /* Contiguous elements after axis  */
    uint32_t index;                 /* Index to gather along the axis  */
} gather_fp16_spatz_params_t;

#endif /* GATHER_FP16_SPATZ_PARAMS_H_ */
