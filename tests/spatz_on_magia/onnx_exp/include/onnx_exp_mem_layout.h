#ifndef ONNX_EXP_H_
#define ONNX_EXP_H_

#include "data.h"
#include "magia_tile_utils.h"
#include "magia_utils.h"
#include "onnx_exp_params.h"

#define ALIGNMENT   (4)

/* Aligns the given address to 4-byte  */
#define ALIGN_4B(addr)  (((addr) + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1))

/* Division with round upwards */
#define DIV_UP(a, b)    (((a) + (b) - 1) / (b))

#define L1_BASE_TILE    (L1_BASE + (get_hartid() * L1_TILE_OFFSET))
#define TENSOR_LEN      (BATCH * CHANNELS * HEIGHT * WIDTH)
#define IN_CHUNK_ELEMS  DIV_UP(TENSOR_LEN, NUM_HARTS)
#define IN_CHUNK_SIZE   (IN_CHUNK_ELEMS * sizeof(float16))
#define OUT_CHUNK_SIZE  (IN_CHUNK_SIZE)
#define SCALAR_SIZE     (sizeof(float16))

#define ONNX_EXP_PARAMS_BASE    (L1_BASE_TILE)
#define ONNX_EXP_PARAMS_SIZE    ALIGN_4B(sizeof(onnx_exp_params_t))

#define INPUT_BASE  ALIGN_4B(ONNX_EXP_PARAMS_BASE + ONNX_EXP_PARAMS_SIZE)
#define INPUT_SIZE  ALIGN_4B(IN_CHUNK_SIZE)

#define RES_BASE    ALIGN_4B(INPUT_BASE + INPUT_SIZE)
#define RES_SIZE    ALIGN_4B(OUT_CHUNK_SIZE)

#define EXP_BASE    ALIGN_4B(RES_BASE + RES_SIZE)
#define EXP_SIZE    ALIGN_4B(OUT_CHUNK_SIZE)

#endif  /* ONNX_EXP_H_ */
