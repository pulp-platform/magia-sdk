#ifndef SWIGLU_FP16_SPATZ_PARAMS_H_
#define SWIGLU_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_G; /* Tile's gate tensor shard                    */
    uintptr_t shard_U; /* Tile's up tensor shard                      */
    uintptr_t shard_Y; /* Tile's output tensor shard (may alias G, U) */

    uint32_t total_elems; /* Total elements in the tensor */
    uint32_t elem_start;  /* Tile's element shard start   */
    uint32_t elem_len;    /* Tile's number of elements    */
} swiglu_fp16_spatz_params_t;

#endif /* SWIGLU_FP16_SPATZ_PARAMS_H_ */
