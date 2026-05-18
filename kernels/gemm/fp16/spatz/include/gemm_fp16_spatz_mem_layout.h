#ifndef GEMM_FP16_SPATZ_MEM_LAYOUT_H_
#define GEMM_FP16_SPATZ_MEM_LAYOUT_H_

#include "data.h"
#include "kernels_mem_layout_utils.h"
#include "magia_tile_utils.h"
#include "magia_utils.h"
#include "gemm_fp16_spatz_params.h"

#define DIM_M               (INPUT0_DIM0)
#define DIM_N               (INPUT1_DIM0)
#define DIM_K               (INPUT0_DIM1)

#define L1_BASE_TILE        (L1_BASE + (get_hartid() * L1_TILE_OFFSET))
#define ROWS_PER_TILE       DIV_UP(DIM_M, NUM_HARTS)

#define SHARD_A_LEN         (ROWS_PER_TILE * DIM_K)
#define SHARD_B_LEN         (DIM_K * DIM_N)
#define SHARD_C_LEN         (ROWS_PER_TILE * DIM_N)
#define SHARD_Y_LEN         (ROWS_PER_TILE * DIM_N)
#define SCALAR_SIZE         (sizeof(float16))

#define GEMM_FP16_SPATZ_PARAMS_BASE     (L1_BASE_TILE)
#define GEMM_FP16_SPATZ_PARAMS_SIZE     ALIGN_4B(sizeof(gemm_fp16_spatz_params_t))

#define SHARD_A_BASE    ALIGN_4B(GEMM_FP16_SPATZ_PARAMS_BASE + GEMM_FP16_SPATZ_PARAMS_SIZE)
#define SHARD_A_SIZE    ALIGN_4B(SHARD_A_LEN * sizeof(float16))

#define SHARD_B_BASE    ALIGN_4B(SHARD_A_BASE + SHARD_A_SIZE)
#define SHARD_B_SIZE    ALIGN_4B(SHARD_B_LEN * sizeof(float16))

#define SHARD_C_BASE    ALIGN_4B(SHARD_B_BASE + SHARD_B_SIZE)
#define SHARD_C_SIZE    ALIGN_4B(SHARD_C_LEN * sizeof(float16))

#define SHARD_Y_BASE    ALIGN_4B(SHARD_C_BASE + SHARD_C_SIZE)
#define SHARD_Y_SIZE    ALIGN_4B(SHARD_Y_LEN * sizeof(float16))

#define ALPHA_BASE      ALIGN_4B(SHARD_Y_BASE + SHARD_Y_SIZE)
#define ALPHA_SIZE      ALIGN_4B(SCALAR_SIZE)

#define BETA_BASE       ALIGN_4B(ALPHA_BASE + ALPHA_SIZE)
#define BETA_SIZE       ALIGN_4B(SCALAR_SIZE)

#endif  /* GEMM_FP16_SPATZ_MEM_LAYOUT_H_ */
