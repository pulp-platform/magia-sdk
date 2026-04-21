#ifndef ONNX_SOFTMAX_H_
#define ONNX_SOFTMAX_H_

#include "data.h"
#include "tile.h"
#include "onnx_softmax_params.h"

#define ALIGNMENT   (4)
/* Aligns the given address to 4-bytes */
#define ALIGN_4B(addr)  (((addr) + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1))

#define L1_BASE_TILE    (L1_BASE + (get_hartid() * L1_TILE_OFFSET))
#define VEC_SIZE        ((LEN) * sizeof(float16))

#define ONNX_SOFTMAX_PARAMS_BASE    (L1_BASE_TILE)
#define ONNX_SOFTMAX_PARAMS_SIZE    ALIGN_4B(sizeof(onnx_softmax_params_t))

#define INPUT_BASE  ALIGN_4B(ONNX_SOFTMAX_PARAMS_BASE + ONNX_SOFTMAX_PARAMS_SIZE)
#define INPUT_SIZE  ALIGN_4B(VEC_SIZE)

#define RES_BASE    ALIGN_4B(INPUT_BASE + INPUT_SIZE)
#define RES_SIZE    ALIGN_4B(VEC_SIZE)

#define EXP_BASE    ALIGN_4B(RES_BASE + RES_SIZE)
#define EXP_SIZE    ALIGN_4B(VEC_SIZE)

#endif  /* ONNX_SOFTMAX_H_ */
