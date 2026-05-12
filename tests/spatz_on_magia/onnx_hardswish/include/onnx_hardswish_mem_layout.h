#ifndef ONNX_HARDSWISH_H_
#define ONNX_HARDSWISH_H_

#include "data.h"
#include "magia_tile_utils.h"
#include "magia_utils.h"
#include "onnx_hardswish_params.h"

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

#define ONNX_HARDSWISH_PARAMS_BASE    (L1_BASE_TILE)
#define ONNX_HARDSWISH_PARAMS_SIZE    ALIGN_4B(sizeof(onnx_hardswish_params_t))

#define CHUNK_X_BASE    ALIGN_4B(ONNX_HARDSWISH_PARAMS_BASE + ONNX_HARDSWISH_PARAMS_SIZE)
#define CHUNK_X_SIZE    ALIGN_4B(IN_CHUNK_SIZE)

#define CHUNK_Y_BASE    ALIGN_4B(CHUNK_X_BASE + CHUNK_X_SIZE)
#define CHUNK_Y_SIZE    ALIGN_4B(OUT_CHUNK_SIZE)

#define CHUNK_G_BASE    ALIGN_4B(CHUNK_Y_BASE + CHUNK_Y_SIZE)
#define CHUNK_G_SIZE    ALIGN_4B(OUT_CHUNK_SIZE)

#endif  /* ONNX_HARDSWISH_H_ */
