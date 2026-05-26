#ifndef SOFTMAX_FP16_SPATZ_MEM_LAYOUT_H_
#define SOFTMAX_FP16_SPATZ_MEM_LAYOUT_H_

#include "data.h"
#include "kernels_mem_layout_utils.h"
#include "magia_tile_utils.h"
#include "magia_utils.h"
#include "softmax_fp16_spatz_params.h"

#define L1_BASE_TILE        (L1_BASE + (get_hartid() * L1_TILE_OFFSET))
#define TOTAL_ROWS          (INPUT0_DIM0 * INPUT0_DIM1 * INPUT0_DIM2)
#define ROWS_PER_TILE       DIV_UP(TOTAL_ROWS, NUM_HARTS)
#define ROW_LEN             (INPUT0_DIM3)
#define SHARD_LEN           (ROWS_PER_TILE * ROW_LEN)
#define SHARD_SIZE          (SHARD_LEN * sizeof(float16))

#define SOFTMAX_FP16_SPATZ_PARAMS_BASE   (L1_BASE_TILE)
#define SOFTMAX_FP16_SPATZ_PARAMS_SIZE   ALIGN_4B(sizeof(softmax_fp16_spatz_params_t))

#define SHARD_INPUT_BASE    ALIGN_4B(SOFTMAX_FP16_SPATZ_PARAMS_BASE + SOFTMAX_FP16_SPATZ_PARAMS_SIZE)
#define SHARD_INPUT_SIZE    ALIGN_4B(SHARD_SIZE)

#define SHARD_OUTPUT_BASE    ALIGN_4B(SHARD_INPUT_BASE + SHARD_INPUT_SIZE)
#define SHARD_OUTPUT_SIZE    ALIGN_4B(SHARD_SIZE)

#endif  /* SOFTMAX_FP16_SPATZ_MEM_LAYOUT_H_ */
