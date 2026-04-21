#ifndef ONNX_RELU_H_
#define ONNX_RELU_H_

#include "data.h"
#include "tile.h"
#include "onnx_relu_params.h"

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

#define ONNX_RELU_PARAMS_BASE    (L1_BASE_TILE)
#define ONNX_RELU_PARAMS_SIZE    ALIGN_4B(sizeof(onnx_relu_params_t))

#define CHUNK_X_BASE    ALIGN_4B(ONNX_RELU_PARAMS_BASE + ONNX_RELU_PARAMS_SIZE)
#define CHUNK_X_SIZE    ALIGN_4B(IN_CHUNK_SIZE)

#define CHUNK_Y_BASE    ALIGN_4B(CHUNK_X_BASE + CHUNK_X_SIZE)
#define CHUNK_Y_SIZE    ALIGN_4B(OUT_CHUNK_SIZE)

#define CHUNK_G_BASE    ALIGN_4B(CHUNK_Y_BASE + CHUNK_Y_SIZE)
#define CHUNK_G_SIZE    ALIGN_4B(OUT_CHUNK_SIZE)


#endif  /* ONNX_RELU_H_ */
