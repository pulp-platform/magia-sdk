#ifndef SUB_FP16_MEM_LAYOUT_H_
#define SUB_FP16_MEM_LAYOUT_H_

#include "data.h"
#include "kernels_mem_layout_utils.h"
#include "magia_tile_utils.h"
#include "magia_utils.h"
#include "sub_fp16_spatz_params.h"

#define L1_BASE_TILE    (L1_BASE + (get_hartid() * L1_TILE_OFFSET))
#define TENSOR_LEN      (INPUT0_SIZE)
#define IN_SHARD_ELEMS  DIV_UP(TENSOR_LEN, NUM_HARTS)
#define IN_SHARD_SIZE   (IN_SHARD_ELEMS * sizeof(float16))
#define OUT_SHARD_SIZE  (IN_SHARD_SIZE)

#define SUB_FP16_SPATZ_PARAMS_BASE    (L1_BASE_TILE)
#define SUB_FP16_SPATZ_PARAMS_SIZE    ALIGN_4B(sizeof(sub_fp16_spatz_params_t))

#define SHARD_A_BASE    ALIGN_4B(SUB_FP16_SPATZ_PARAMS_BASE + SUB_FP16_SPATZ_PARAMS_SIZE)
#define SHARD_A_SIZE    ALIGN_4B(IN_SHARD_SIZE)

#define SHARD_B_BASE    ALIGN_4B(SHARD_A_BASE + SHARD_A_SIZE)
#define SHARD_B_SIZE    ALIGN_4B(IN_SHARD_SIZE)

#define SHARD_C_BASE    ALIGN_4B(SHARD_B_BASE + SHARD_B_SIZE)
#define SHARD_C_SIZE    ALIGN_4B(OUT_SHARD_SIZE)

#endif  /* SUB_FP16_MEM_LAYOUT_H_ */
