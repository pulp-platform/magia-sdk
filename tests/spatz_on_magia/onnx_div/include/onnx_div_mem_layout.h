#ifndef ONNX_DIV_H_
#define ONNX_DIV_H_

#include "data.h"
#include "magia_tile_utils.h"
#include "magia_utils.h"
#include "onnx_div_params.h"

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

#define ONNX_DIV_PARAMS_BASE    (L1_BASE_TILE)
#define ONNX_DIV_PARAMS_SIZE    ALIGN_4B(sizeof(onnx_div_params_t))

#define CHUNK_A_BASE    ALIGN_4B(ONNX_DIV_PARAMS_BASE + ONNX_DIV_PARAMS_SIZE)
#define CHUNK_A_SIZE    ALIGN_4B(IN_CHUNK_SIZE)

#define CHUNK_B_BASE    ALIGN_4B(CHUNK_A_BASE + CHUNK_A_SIZE)
#define CHUNK_B_SIZE    ALIGN_4B(IN_CHUNK_SIZE)

#define CHUNK_C_BASE    ALIGN_4B(CHUNK_B_BASE + CHUNK_B_SIZE)
#define CHUNK_C_SIZE    ALIGN_4B(OUT_CHUNK_SIZE)

#define CHUNK_G_BASE    ALIGN_4B(CHUNK_C_BASE + CHUNK_C_SIZE)
#define CHUNK_G_SIZE    ALIGN_4B(OUT_CHUNK_SIZE)

#endif  /* ONNX_DIV_H_ */
