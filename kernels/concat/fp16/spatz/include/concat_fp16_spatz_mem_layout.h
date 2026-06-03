#ifndef CONCAT_FP16_SPATZ_MEM_LAYOUT_H_
#define CONCAT_FP16_SPATZ_MEM_LAYOUT_H_

#include "data.h"
#include "kernels_mem_layout_utils.h"
#include "magia_tile_utils.h"
#include "magia_utils.h"
#include "concat_fp16_spatz_params.h"

/* Weak as hell - if two tensors has same shape axis inference will fail */
#if (INPUT0_DIM0 != INPUT1_DIM0)
    #error "Axis 0 concatenation is not supported on Spatz kernel! Handle via flat L2 DMA copy."
#elif (INPUT0_DIM1 != INPUT1_DIM1)
    #define SHARD_ITERATION DIV_UP(INPUT0_DIM0, NUM_HARTS)
    #define SHARD_IN0_ELEMS (INPUT0_DIM1 * INPUT0_DIM2 * INPUT0_DIM3)
    #define SHARD_IN1_ELEMS (INPUT1_DIM1 * INPUT1_DIM2 * INPUT1_DIM3)
#elif (INPUT0_DIM2 != INPUT1_DIM2)
    #define SHARD_ITERATION DIV_UP(INPUT0_DIM0 * INPUT0_DIM1, NUM_HARTS)
    #define SHARD_IN0_ELEMS (INPUT0_DIM2 * INPUT0_DIM3)
    #define SHARD_IN1_ELEMS (INPUT1_DIM2 * INPUT1_DIM3)
#else
    /* Fallback to axis = -1 (axis = 3) */
    #define SHARD_ITERATION DIV_UP(INPUT0_DIM0 * INPUT0_DIM1 * INPUT0_DIM2, NUM_HARTS)
    #define SHARD_IN0_ELEMS (INPUT0_DIM3)
    #define SHARD_IN1_ELEMS (INPUT1_DIM3)
#endif

#define SHARD_OUT_ELEMS     (SHARD_IN0_ELEMS + SHARD_IN1_ELEMS)

#define L1_BASE_TILE                    (L1_BASE + (get_hartid() * L1_TILE_OFFSET))
#define CONCAT_FP16_SPATZ_PARAMS_BASE   (L1_BASE_TILE)
#define CONCAT_FP16_SPATZ_PARAMS_SIZE   ALIGN_4B(sizeof(concat_fp16_spatz_params_t))

#define SHARD_INPUT0_BASE    ALIGN_4B(CONCAT_FP16_SPATZ_PARAMS_BASE + CONCAT_FP16_SPATZ_PARAMS_SIZE)
#define SHARD_INPUT0_SIZE    ALIGN_4B(SHARD_ITERATION * SHARD_IN0_ELEMS * sizeof(float16))

#define SHARD_INPUT1_BASE    ALIGN_4B(SHARD_INPUT0_BASE + SHARD_INPUT0_SIZE)
#define SHARD_INPUT1_SIZE    ALIGN_4B(SHARD_ITERATION * SHARD_IN1_ELEMS * sizeof(float16))

#define SHARD_OUTPUT_BASE   ALIGN_4B(SHARD_INPUT1_BASE + SHARD_INPUT1_SIZE)
#define SHARD_OUTPUT_SIZE   ALIGN_4B(SHARD_ITERATION * SHARD_OUT_ELEMS * sizeof(float16))

#endif  /* CONCAT_FP16_SPATZ_MEM_LAYOUT_H_ */
