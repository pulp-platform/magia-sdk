#ifndef ONNX_CLIP_MEM_LAYOUT_H_
#define ONNX_CLIP_MEM_LAYOUT_H_

#include "magia_utils.h"
#include "data.h"
#include "onnx_clip_params.h"

#define ALIGNMENT   4
/* Aligns the given address to 4-byte  */
#define ALIGN_4B(addr)  (((addr) + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1))

#define L1_BASE_TILE    (L1_BASE + (get_hartid() * L1_TILE_OFFSET))
#define VEC_SIZE        (LEN * sizeof(float16))
#define SCALAR_SIZE     (sizeof(float16))

#define ONNX_CLIP_PARAMS_BASE    L1_BASE_TILE
#define ONNX_CLIP_PARAMS_SIZE    ALIGN_4B(sizeof(onnx_clip_params_t))

#define INPUT_VEC_BASE  ALIGN_4B(ONNX_CLIP_PARAMS_BASE + ONNX_CLIP_PARAMS_SIZE)
#define INPUT_VEC_SIZE  ALIGN_4B(VEC_SIZE)

#define MIN_VAL_BASE    ALIGN_4B(INPUT_VEC_BASE + INPUT_VEC_SIZE)
#define MIN_VAL_SIZE    ALIGN_4B(SCALAR_SIZE)

#define MAX_VAL_BASE    ALIGN_4B(MIN_VAL_BASE + MIN_VAL_SIZE)
#define MAX_VAL_SIZE    ALIGN_4B(SCALAR_SIZE)

#define RES_BASE        ALIGN_4B(MAX_VAL_BASE + MAX_VAL_SIZE)
#define RES_SIZE        ALIGN_4B(VEC_SIZE)

#define EXP_BASE        ALIGN_4B(RES_BASE + RES_SIZE)
#define EXP_SIZE        ALIGN_4B(VEC_SIZE)

#endif  /* ONNX_CLIP_MEM_LAYOUT_H_ */
