#ifndef CONV2D_FP16_SPATZ_MEM_LAYOUT_H_
#define CONV2D_FP16_SPATZ_MEM_LAYOUT_H_

#include "data.h"
#include "kernels_mem_layout_utils.h"
#include "magia_tile_utils.h"
#include "magia_utils.h"
#include "conv2d_fp16_spatz_params.h"

#define L1_BASE_TILE        (L1_BASE + (get_hartid() * L1_TILE_OFFSET))

#define C_OUT_PER_TILE      DIV_UP(OUTPUT0_DIM1, NUM_HARTS)

#define INPUT_HW_LEN        (INPUT0_DIM2 * INPUT0_DIM3)
#define FULL_INPUT_LEN      (INPUT0_DIM1 * INPUT_HW_LEN)
#define FULL_INPUT_SIZE     (FULL_INPUT_LEN * sizeof(float16))

#define KERNEL_HW_LEN       (INPUT0_DIM2 * INPUT0_DIM3)
#define WEIGHT_SHARD_LEN    (C_OUT_PER_TILE * INPUT0_DIM1 * KERNEL_HW_LEN)
#define WEIGHT_SHARD_SIZE   (WEIGHT_SHARD_LEN * sizeof(float16))

#define OUTPUT_HW_LEN       (OUTPUT0_DIM2 * OUTPUT0_DIM3)
#define OUTPUT_SHARD_LEN    (C_OUT_PER_TILE * OUTPUT_HW_LEN)
#define OUTPUT_SHARD_SIZE   (OUTPUT_SHARD_LEN * sizeof(float16))

#define CONV2D_FP16_SPATZ_PARAMS_BASE    (L1_BASE_TILE)
#define CONV2D_FP16_SPATZ_PARAMS_SIZE    ALIGN_4B(sizeof(conv2d_fp16_spatz_params_t))

#define SHARD_X_BASE    ALIGN_4B(CONV2D_FP16_SPATZ_PARAMS_BASE + CONV2D_FP16_SPATZ_PARAMS_SIZE)
#define SHARD_X_SIZE    ALIGN_4B(FULL_INPUT_SIZE)

#define SHARD_W_BASE    ALIGN_4B(SHARD_X_BASE + SHARD_X_SIZE)
#define SHARD_W_SIZE    ALIGN_4B(WEIGHT_SHARD_SIZE)

#define SHARD_Y_BASE    ALIGN_4B(SHARD_W_BASE + SHARD_W_SIZE)
#define SHARD_Y_SIZE    ALIGN_4B(OUTPUT_SHARD_SIZE)

#endif  /* CONV2D_FP16_SPATZ_MEM_LAYOUT_H_ */
