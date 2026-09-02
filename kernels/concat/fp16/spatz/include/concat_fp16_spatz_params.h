#ifndef CONCAT_FP16_SPATZ_PARAMS_H_
#define CONCAT_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_input;      /* -> L1 array of num_inputs tensor shard addresses */
    uintptr_t len_input;        /* -> L1 array of num_inputs per-iteration lengths  */
    uintptr_t shard_output;     /* Tile's output tensor shard                       */

    uint32_t iter_start;        /* Tile's iteration shard start                     */
    uint32_t iter_len;          /* Tile's number of iterations                      */
    uint32_t num_inputs;        /* Number of concatenated inputs                    */
} concat_fp16_spatz_params_t;

#endif  /* CONCAT_FP16_SPATZ_PARAMS_H_ */
