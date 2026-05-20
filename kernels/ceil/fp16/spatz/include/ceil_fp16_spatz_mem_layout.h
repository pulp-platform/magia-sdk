#ifndef CEIL_FP16_SPATZ_MEM_LAYOUT_H_
#define CEIL_FP16_SPATZ_MEM_LAYOUT_H_

#include "data.h"
#include "kernels_mem_layout_utils.h"
#include "magia_tile_utils.h"
#include "magia_utils.h"
#include "ceil_fp16_spatz_params.h"

#define L1_BASE_TILE    (L1_BASE + (get_hartid() * L1_TILE_OFFSET))
#define TENSOR_LEN      (INPUT0_SIZE)
#define IN_SHARD_ELEMS  DIV_UP(TENSOR_LEN, NUM_HARTS)
#define IN_SHARD_SIZE   (IN_SHARD_ELEMS * sizeof(float16))
#define OUT_SHARD_SIZE  (IN_SHARD_SIZE)

#define CEIL_FP16_SPATZ_PARAMS_BASE    (L1_BASE_TILE)
#define CEIL_FP16_SPATZ_PARAMS_SIZE    ALIGN_4B(sizeof(ceil_fp16_spatz_params_t))

#define SHARD_X_BASE    ALIGN_4B(CEIL_FP16_SPATZ_PARAMS_BASE + CEIL_FP16_SPATZ_PARAMS_SIZE)
#define SHARD_X_SIZE    ALIGN_4B(IN_SHARD_SIZE)

#define SHARD_Y_BASE    ALIGN_4B(SHARD_X_BASE + SHARD_X_SIZE)
#define SHARD_Y_SIZE    ALIGN_4B(IN_SHARD_SIZE)

#endif  /* CEIL_FP16_SPATZ_MEM_LAYOUT_H_ */
