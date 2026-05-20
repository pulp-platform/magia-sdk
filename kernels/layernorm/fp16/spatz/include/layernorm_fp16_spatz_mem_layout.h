#ifndef LAYERNORM_FP16_SPATZ_MEM_LAYOUT_H_
#define LAYERNORM_FP16_SPATZ_MEM_LAYOUT_H_

#include "data.h"
#include "kernels_mem_layout_utils.h"
#include "magia_tile_utils.h"
#include "magia_utils.h"
#include "layernorm_fp16_spatz_params.h"

#define L1_BASE_TILE        (L1_BASE + (get_hartid() * L1_TILE_OFFSET))
#define TOTAL_ROWS          (INPUT0_DIM0 * INPUT0_DIM1 * INPUT0_DIM2)
#define ROWS_PER_TILE       DIV_UP(TOTAL_ROWS, NUM_HARTS)
#define ROW_LEN             (INPUT0_DIM3)
#define SHARD_LEN           (ROWS_PER_TILE * ROW_LEN)
#define SHARD_SIZE          (SHARD_LEN * sizeof(float16))
#define PARAMS_SIZE         (ROW_LEN * sizeof(float16))
#define SCALAR_SIZE         (sizeof(float16))

#define LAYERNORM_FP16_SPATZ_PARAMS_BASE   (L1_BASE_TILE)
#define LAYERNORM_FP16_SPATZ_PARAMS_SIZE   ALIGN_4B(sizeof(layernorm_fp16_spatz_params_t))

#define SHARD_X_BASE    ALIGN_4B(LAYERNORM_FP16_SPATZ_PARAMS_BASE + LAYERNORM_FP16_SPATZ_PARAMS_SIZE)
#define SHARD_X_SIZE    ALIGN_4B(SHARD_SIZE)

#define SHARD_Y_BASE    ALIGN_4B(SHARD_X_BASE + SHARD_X_SIZE)
#define SHARD_Y_SIZE    ALIGN_4B(SHARD_SIZE)

#define GAMMA_BASE      ALIGN_4B(SHARD_Y_BASE + SHARD_Y_SIZE)
#define GAMMA_SIZE      ALIGN_4B(PARAMS_SIZE)

#define BETA_BASE       ALIGN_4B(GAMMA_BASE + GAMMA_SIZE)
#define BETA_SIZE       ALIGN_4B(PARAMS_SIZE)

#define EPS_BASE        ALIGN_4B(BETA_BASE + BETA_SIZE)
#define EPS_SIZE        ALIGN_4B(SCALAR_SIZE)

#endif  /* LAYERNORM_FP16_SPATZ_MEM_LAYOUT_H_ */
