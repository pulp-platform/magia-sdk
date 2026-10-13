#ifndef MAXPOOL2D_FP16_SPATZ_MEM_LAYOUT_H_
#define MAXPOOL2D_FP16_SPATZ_MEM_LAYOUT_H_

#include "data.h"
#include "kernels_mem_layout_utils.h"
#include "magia_tile_utils.h"
#include "magia_utils.h"
#include "maxpool2d_fp16_spatz_params.h"

#define L1_BASE_TILE        (L1_BASE + (get_hartid() * L1_TILE_OFFSET))
#define CHANNELS_PER_TILE   DIV_UP(INPUT0_DIM1, NUM_HARTS)
#define INPUT_HW_LEN        (INPUT0_DIM2 * INPUT0_DIM3)
#define OUTPUT_HW_LEN       (OUTPUT0_DIM2 * OUTPUT0_DIM3)
#define INPUT_LEN           (CHANNELS_PER_TILE * INPUT_HW_LEN)
#define INPUT_SIZE          (INPUT_LEN * sizeof(float16))
#define OUTPUT_LEN          (CHANNELS_PER_TILE * OUTPUT_HW_LEN)
#define OUTPUT_SIZE         (OUTPUT_LEN * sizeof(float16))

#define MAXPOOL2D_FP16_SPATZ_PARAMS_BASE    (L1_BASE_TILE)
#define MAXPOOL2D_FP16_SPATZ_PARAMS_SIZE    ALIGN_4B(sizeof(maxpool2d_fp16_spatz_params_t))

#define SHARD_X_BASE        ALIGN_4B(MAXPOOL2D_FP16_SPATZ_PARAMS_BASE + MAXPOOL2D_FP16_SPATZ_PARAMS_SIZE)
#define SHARD_X_SIZE        ALIGN_4B(INPUT_SIZE)

#define SHARD_Y_BASE        ALIGN_4B(SHARD_X_BASE + SHARD_X_SIZE)
#define SHARD_Y_SIZE        ALIGN_4B(OUTPUT_SIZE)


#endif  /* MAXPOOL2D_FP16_SPATZ_MEM_LAYOUT_H_ */
