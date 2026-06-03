#ifndef CONCAT_FP16_SPATZ_PARAMS_H_
#define CONCAT_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_input0;         /* Tile's input0 tensor shard       */
    uintptr_t shard_input1;         /* Tile's input1 tensor shard       */
    uintptr_t shard_output;         /* Tile's output tensor shard       */

    uint32_t iter_start;            /* Tile's iteration shard start     */
    uint32_t iter_len;              /* Tile's number of iterations      */
    uint32_t len_input0;            /* Elements of Input0 per iteration */
    uint32_t len_input1;            /* Elements of Input1 per iteration */
} concat_fp16_spatz_params_t;

#endif  /* CONCAT_FP16_SPATZ_PARAMS_H_ */
