#ifndef GATHER_FP16_SPATZ_MEM_LAYOUT_H_
#define GATHER_FP16_SPATZ_MEM_LAYOUT_H_

#include "data.h"
#include "kernels_mem_layout_utils.h"
#include "magia_tile_utils.h"
#include "magia_utils.h"
#include "gather_fp16_spatz_params.h"

#if (INPUT0_DIM0 != OUTPUT0_DIM0)
    #error "Axis 0 gather is not supported on Spatz kernel! Handle via flat L2 DMA copy."
#elif (INPUT0_DIM1 != OUTPUT0_DIM1)
    #define SHARD_ITERATION DIV_UP(INPUT0_DIM0, NUM_HARTS)
    #define SHARD_IN_ELEMS  (INPUT0_DIM1 * INPUT0_DIM2 * INPUT0_DIM3)
    #define SHARD_OUT_ELEMS (OUTPUT0_DIM1 * OUTPUT0_DIM2 * OUTPUT0_DIM3)
#elif (INPUT0_DIM2 != OUTPUT0_DIM2)
    #define SHARD_ITERATION DIV_UP(INPUT0_DIM0 * INPUT0_DIM1, NUM_HARTS)
    #define SHARD_IN_ELEMS  (INPUT0_DIM2 * INPUT0_DIM3)
    #define SHARD_OUT_ELEMS (OUTPUT0_DIM2 * OUTPUT0_DIM3)
#else
    /* Fallback to axis = -1 (axis = 3) */
    #define SHARD_ITERATION DIV_UP(INPUT0_DIM0 * INPUT0_DIM1 * INPUT0_DIM2, NUM_HARTS)
    #define SHARD_IN_ELEMS  (INPUT0_DIM3)
    #define SHARD_OUT_ELEMS (OUTPUT0_DIM3)
#endif

#define L1_BASE_TILE                    (L1_BASE + (get_hartid() * L1_TILE_OFFSET))
#define GATHER_FP16_SPATZ_PARAMS_BASE   (L1_BASE_TILE)
#define GATHER_FP16_SPATZ_PARAMS_SIZE   ALIGN_4B(sizeof(gather_fp16_spatz_params_t))

#define SHARD_INPUT_BASE    ALIGN_4B(GATHER_FP16_SPATZ_PARAMS_BASE + GATHER_FP16_SPATZ_PARAMS_SIZE)
#define SHARD_INPUT_SIZE    ALIGN_4B(SHARD_ITERATION * SHARD_IN_ELEMS * sizeof(float16))

#define SHARD_OUTPUT_BASE   ALIGN_4B(SHARD_INPUT_BASE + SHARD_INPUT_SIZE)
#define SHARD_OUTPUT_SIZE   ALIGN_4B(SHARD_ITERATION * SHARD_OUT_ELEMS * sizeof(float16))

#endif  /* GATHER_FP16_SPATZ_MEM_LAYOUT_H_ */
