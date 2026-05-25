#ifndef GLOBALAVERAGEPOOL_FP16_SPATZ_MEM_LAYOUT_H_
#define GLOBALAVERAGEPOOL_FP16_SPATZ_MEM_LAYOUT_H_

#include "data.h"
#include "kernels_mem_layout_utils.h"
#include "magia_tile_utils.h"
#include "magia_utils.h"
#include "globalaveragepool_fp16_spatz_params.h"

#define L1_BASE_TILE        (L1_BASE + (get_hartid() * L1_TILE_OFFSET))
#define CHANNEL_NUM         (INPUT0_DIM0 * INPUT0_DIM1)
#define CHANNELS_PER_TILE   DIV_UP(CHANNEL_NUM, NUM_HARTS)
#define CHANNEL_LEN         (INPUT0_DIM2 * INPUT0_DIM3)
#define SHARD_IN_LEN        (CHANNELS_PER_TILE * CHANNEL_LEN)
#define SHARD_IN_SIZE       (SHARD_IN_LEN * sizeof(float16))
#define SHARD_OUT_LEN       (CHANNELS_PER_TILE)
#define SHARD_OUT_SIZE      (SHARD_OUT_LEN * sizeof(float16))

#define GLOBALAVERAGEPOOL_FP16_SPATZ_PARAMS_BASE   (L1_BASE_TILE)
#define GLOBALAVERAGEPOOL_FP16_SPATZ_PARAMS_SIZE   ALIGN_4B(sizeof(globalaveragepool_fp16_spatz_params_t))

#define SHARD_X_BASE    ALIGN_4B(GLOBALAVERAGEPOOL_FP16_SPATZ_PARAMS_BASE + GLOBALAVERAGEPOOL_FP16_SPATZ_PARAMS_SIZE)
#define SHARD_X_SIZE    ALIGN_4B(SHARD_IN_SIZE)

#define SHARD_Y_BASE    ALIGN_4B(SHARD_X_BASE + SHARD_X_SIZE)
#define SHARD_Y_SIZE    ALIGN_4B(SHARD_OUT_SIZE)

#endif  /* GLOBALAVERAGEPOOL_FP16_SPATZ_MEM_LAYOUT_H_ */
