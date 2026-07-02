#ifndef SLICE_FP16_SPATZ_PARAMS_H_
#define SLICE_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X;          /* Tile's input tensor shard        */
    uintptr_t shard_Y;          /* Tile's output tensor shard       */

    uint32_t slice_dim;         /* Original dimension at axis       */
    uint32_t out_slice_dim;     /* Output slice dimension           */
    uint32_t inner_dim;         /* Total inner elements size        */
    uint32_t start_idx;         /* Slice start index at axis        */

    uint32_t start_outer;       /* Start outer iteration for tile   */
    uint32_t len_outer;         /* Number of outer loops for tile   */
} slice_fp16_spatz_params_t;

#endif  /* SLICE_FP16_SPATZ_PARAMS_H_ */
