#ifndef CONVTRANSPOSE_FP16_SPATZ_MEM_LAYOUT_H_
#define CONVTRANSPOSE_FP16_SPATZ_MEM_LAYOUT_H_

#include "data.h"
#include "kernels_mem_layout_utils.h"
#include "magia_tile_utils.h"
#include "magia_utils.h"
#include "convtranspose_fp16_spatz_params.h"

#define L1_BASE_TILE            (L1_BASE + (get_hartid() * L1_TILE_OFFSET))
#define C_OUT_PER_TILE          DIV_UP(OUTPUT0_DIM1, NUM_HARTS)
#define MAX_C_IN_PER_TILE       (INPUT0_DIM1)   // Worst Case: G = 1
#define INPUT_HW_LEN            (INPUT0_DIM2 * INPUT0_DIM3)
#define CONV_INPUT_LEN          (INPUT0_DIM0 * MAX_C_IN_PER_TILE * INPUT_HW_LEN)
#define INPUT_SIZE              (CONV_INPUT_LEN * sizeof(float16))
#define WEIGTHS_HW_LEN          (INPUT1_DIM2 * INPUT1_DIM3)
#define CONV_KERNEL_LEN         (MAX_C_IN_PER_TILE * C_OUT_PER_TILE * WEIGTHS_HW_LEN)
#define WEIGTHS_HW_SIZE         (CONV_KERNEL_LEN * sizeof(float16))
#define OUTPUT_HW_LEN           (OUTPUT0_DIM2 * OUTPUT0_DIM3)
#define CONV_OUTPUT_LEN         (INPUT0_DIM0 * C_OUT_PER_TILE * OUTPUT_HW_LEN)
#define OUTPUT_SIZE             (CONV_OUTPUT_LEN * sizeof(float16))

// #define CONVTRANSPOSE_FP16_SPATZ_PARAMS_BASE   (L1_BASE_TILE)
// #define CONVTRANSPOSE_FP16_SPATZ_PARAMS_SIZE   ALIGN_4B(sizeof(convtranspose_fp16_spatz_params_t))

// #define SHARD_X_BASE    ALIGN_4B(CONVTRANSPOSE_FP16_SPATZ_PARAMS_BASE + CONVTRANSPOSE_FP16_SPATZ_PARAMS_SIZE)
// #define SHARD_X_SIZE    ALIGN_4B(INPUT_SIZE)

// #define SHARD_W_BASE    ALIGN_4B(SHARD_X_BASE + SHARD_X_SIZE)
// #define SHARD_W_SIZE    ALIGN_4B(WEIGTHS_HW_SIZE)

// #define SHARD_Y_BASE    ALIGN_4B(SHARD_W_BASE + SHARD_W_SIZE)
// #define SHARD_Y_SIZE    ALIGN_4B(OUTPUT_SIZE)

#endif  /* CONVTRANSPOSE_FP16_SPATZ_MEM_LAYOUT_H_ */
