#ifndef TRANSPOSE_FP16_SPATZ_MEM_LAYOUT_H_
#define TRANSPOSE_FP16_SPATZ_MEM_LAYOUT_H_

#include "data.h"
#include "kernels_mem_layout_utils.h"
#include "magia_tile_utils.h"
#include "magia_utils.h"
#include "transpose_fp16_spatz_params.h"

/* Shard on first output dim */
#define SHARD_ITERATION                 DIV_UP(OUTPUT0_DIM0, NUM_HARTS)

#define SHARD_IN0_ELEMS                 (INPUT0_SIZE / INPUT0_DIM0)
#define SHARD_OUT_ELEMS                 (OUTPUT0_SIZE / OUTPUT0_DIM0)

/* TODO: remove */
#define STATIC_IN_STRIDE3               (1)
#define STATIC_IN_STRIDE2               (INPUT0_DIM3)
#define STATIC_IN_STRIDE1               (INPUT0_DIM2 * INPUT0_DIM3)
#define STATIC_IN_STRIDE0               (INPUT0_DIM1 * INPUT0_DIM2 * INPUT0_DIM3)

#define L1_BASE_TILE                        (L1_BASE + (get_hartid() * L1_TILE_OFFSET))
#define TRANSPOSE_FP16_SPATZ_PARAMS_BASE    (L1_BASE_TILE)
#define TRANSPOSE_FP16_SPATZ_PARAMS_SIZE    ALIGN_4B(sizeof(transpose_fp16_spatz_params_t))

#define SHARD_INPUT_BASE    ALIGN_4B(TRANSPOSE_FP16_SPATZ_PARAMS_BASE + TRANSPOSE_FP16_SPATZ_PARAMS_SIZE)
#define SHARD_INPUT_SIZE    ALIGN_4B(SHARD_ITERATION * SHARD_IN0_ELEMS * sizeof(float16))

#define SHARD_OUTPUT_BASE   ALIGN_4B(SHARD_INPUT_BASE + SHARD_INPUT_SIZE)
#define SHARD_OUTPUT_SIZE   ALIGN_4B(SHARD_ITERATION * SHARD_OUT_ELEMS * sizeof(float16))

#endif  /* TRANSPOSE_FP16_SPATZ_MEM_LAYOUT_H_ */
