#ifndef MATMUL_FP16_SPATZ_MEM_LAYOUT_H_
#define MATMUL_FP16_SPATZ_MEM_LAYOUT_H_

#include "data.h"
#include "kernels_mem_layout_utils.h"
#include "magia_tile_utils.h"
#include "magia_utils.h"
#include "matmul_fp16_spatz_params.h"


#define L1_BASE_TILE        (L1_BASE + (get_hartid() * L1_TILE_OFFSET))

#define TOTAL_BATCHES       (INPUT0_DIM0 * INPUT0_DIM1)
#define MAX_BATCH_PER_CORE  ((TOTAL_BATCHES / NUM_HARTS) + ((TOTAL_BATCHES % NUM_HARTS) > 0 ? 1 : 0))
#define MAT_A_2D_LEN        (INPUT0_DIM2 * INPUT0_DIM3)
#define MAT_B_2D_LEN        (INPUT0_DIM3 * INPUT1_DIM3)
#define MAT_Y_2D_LEN        (INPUT0_DIM2 * INPUT1_DIM3)
#define SHARD_A_LEN         (MAT_A_2D_LEN * MAX_BATCH_PER_CORE)
#define SHARD_B_LEN         (MAT_B_2D_LEN * MAX_BATCH_PER_CORE)
#define SHARD_Y_LEN         (MAT_Y_2D_LEN * MAX_BATCH_PER_CORE)

#define MATMUL_FP16_SPATZ_PARAMS_BASE   (L1_BASE_TILE)
#define MATMUL_FP16_SPATZ_PARAMS_SIZE   ALIGN_4B(sizeof(matmul_fp16_spatz_params_t))

#define SHARD_A_BASE        ALIGN_4B(MATMUL_FP16_SPATZ_PARAMS_BASE + MATMUL_FP16_SPATZ_PARAMS_SIZE)
#define SHARD_A_SIZE        ALIGN_4B(SHARD_A_LEN * sizeof(float16))

#define SHARD_B_BASE        ALIGN_4B(SHARD_A_BASE + SHARD_A_SIZE)
#define SHARD_B_SIZE        ALIGN_4B(SHARD_B_LEN * sizeof(float16))

#define SHARD_Y_BASE        ALIGN_4B(SHARD_B_BASE + SHARD_B_SIZE)
#define SHARD_Y_SIZE        ALIGN_4B(SHARD_Y_LEN * sizeof(float16))

#endif  /* MATMUL_FP16_SPATZ_MEM_LAYOUT_H_ */
