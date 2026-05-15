#ifndef BATCHNORM_FP16_SPATZ_MEM_LAYOUT_H_
#define BATCHNORM_FP16_SPATZ_MEM_LAYOUT_H_

#include "data.h"
#include "kernels_mem_layout_utils.h"
#include "magia_tile_utils.h"
#include "magia_utils.h"
#include "batchnorm_fp16_spatz_params.h"

#define L1_BASE_TILE        (L1_BASE + (get_hartid() * L1_TILE_OFFSET))
#define CHANNELS_PER_TILE   DIV_UP(INPUT0_DIM1, NUM_HARTS)
#define CHANNEL_LEN         (INPUT0_DIM2 * INPUT0_DIM3)
#define SHARD_LEN           (CHANNELS_PER_TILE * CHANNEL_LEN)
#define SHARD_SIZE          (SHARD_LEN * sizeof(float16))
#define PARAMS_SIZE         (INPUT0_DIM1 * sizeof(float16))
#define SCALAR_SIZE         (sizeof(float16))

#define BATCHNORM_FP16_SPATZ_PARAMS_BASE   (L1_BASE_TILE)
#define BATCHNORM_FP16_SPATZ_PARAMS_SIZE   ALIGN_4B(sizeof(batchnorm_fp16_spatz_params_t))

#define SHARD_X_BASE    ALIGN_4B(BATCHNORM_FP16_SPATZ_PARAMS_BASE + BATCHNORM_FP16_SPATZ_PARAMS_SIZE)
#define SHARD_X_SIZE    ALIGN_4B(SHARD_SIZE)

#define SHARD_Y_BASE    ALIGN_4B(SHARD_X_BASE + SHARD_X_SIZE)
#define SHARD_Y_SIZE    ALIGN_4B(SHARD_SIZE)

#define GAMMA_BASE      ALIGN_4B(SHARD_Y_BASE + SHARD_Y_SIZE)
#define GAMMA_SIZE      ALIGN_4B(PARAMS_SIZE)

#define BETA_BASE       ALIGN_4B(GAMMA_BASE + GAMMA_SIZE)
#define BETA_SIZE       ALIGN_4B(PARAMS_SIZE)

#define MEAN_BASE       ALIGN_4B(BETA_BASE + BETA_SIZE)
#define MEAN_SIZE       ALIGN_4B(PARAMS_SIZE)

#define VAR_BASE        ALIGN_4B(MEAN_BASE + MEAN_SIZE)
#define VAR_SIZE        ALIGN_4B(PARAMS_SIZE)

#define EPS_BASE        ALIGN_4B(VAR_BASE + VAR_SIZE)
#define EPS_SIZE        ALIGN_4B(SCALAR_SIZE)

#endif  /* BATCHNORM_FP16_SPATZ_MEM_LAYOUT_H_ */
