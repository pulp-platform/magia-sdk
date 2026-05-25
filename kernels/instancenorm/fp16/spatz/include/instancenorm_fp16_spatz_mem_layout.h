#ifndef INSTANCENORM_FP16_SPATZ_MEM_LAYOUT_H_
#define INSTANCENORM_FP16_SPATZ_MEM_LAYOUT_H_

#include "data.h"
#include "kernels_mem_layout_utils.h"
#include "magia_tile_utils.h"
#include "magia_utils.h"
#include "instancenorm_fp16_spatz_params.h"

#define L1_BASE_TILE         (L1_BASE + (get_hartid() * L1_TILE_OFFSET))
#define TOTAL_INSTANCES      (INPUT0_DIM0 * INPUT0_DIM1)
#define INSTANCES_PER_TILE   DIV_UP(TOTAL_INSTANCES, NUM_HARTS)
#define INSTANCE_LEN         (INPUT0_DIM2 * INPUT0_DIM3)

#define SHARD_LEN            (INSTANCES_PER_TILE * INSTANCE_LEN)
#define SHARD_SIZE           (SHARD_LEN * sizeof(float16))
#define PARAMS_SIZE          (INPUT0_DIM1 * sizeof(float16))
#define SCALAR_SIZE          (sizeof(float16))

#define INSTANCENORM_FP16_SPATZ_PARAMS_BASE   (L1_BASE_TILE)
#define INSTANCENORM_FP16_SPATZ_PARAMS_SIZE   ALIGN_4B(sizeof(instancenorm_fp16_spatz_params_t))

#define SHARD_INPUT_BASE    ALIGN_4B(INSTANCENORM_FP16_SPATZ_PARAMS_BASE + INSTANCENORM_FP16_SPATZ_PARAMS_SIZE)
#define SHARD_INPUT_SIZE    ALIGN_4B(SHARD_SIZE)

#define SHARD_OUTPUT_BASE    ALIGN_4B(SHARD_INPUT_BASE + SHARD_INPUT_SIZE)
#define SHARD_OUTPUT_SIZE    ALIGN_4B(SHARD_SIZE)

#define GAMMA_BASE      ALIGN_4B(SHARD_OUTPUT_BASE + SHARD_OUTPUT_SIZE)
#define GAMMA_SIZE      ALIGN_4B(PARAMS_SIZE)

#define BETA_BASE       ALIGN_4B(GAMMA_BASE + GAMMA_SIZE)
#define BETA_SIZE       ALIGN_4B(PARAMS_SIZE)

#define EPS_BASE        ALIGN_4B(BETA_BASE + BETA_SIZE)
#define EPS_SIZE        ALIGN_4B(SCALAR_SIZE)

#endif  /* INSTANCENORM_FP16_SPATZ_MEM_LAYOUT_H_ */
