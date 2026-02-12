#ifndef ONNX_ADD_H_
#define ONNX_ADD_H_

#include "data.h"
#include "magia_tile_utils.h"
#include "onnx_add_params.h"

#define ALIGNMENT   4
/* Aligns the given address to 4-byte  */
#define ALIGN_4B(addr)  (((addr) + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1))

#define VEC_SIZE    (LEN * sizeof(float16))

#define ONNX_ADD_PARAMS_BASE    L1_BASE
#define ONNX_ADD_PARAMS_SIZE    ALIGN_4B(sizeof(onnx_add_params_t))

#define SRC_A_BASE  ALIGN_4B(ONNX_ADD_PARAMS_BASE + ONNX_ADD_PARAMS_SIZE)
#define SRC_A_SIZE  ALIGN_4B(VEC_SIZE)

#define SRC_B_BASE  ALIGN_4B(SRC_A_BASE + SRC_A_SIZE)
#define SRC_B_SIZE  ALIGN_4B(VEC_SIZE)

#define RES_BASE    ALIGN_4B(SRC_B_BASE + SRC_B_SIZE)
#define RES_SIZE    ALIGN_4B(VEC_SIZE)

#define EXP_BASE    ALIGN_4B(RES_BASE + RES_SIZE)
#define EXP_SIZE    ALIGN_4B(VEC_SIZE)

#endif  /* ONNX_ADD_H_ */
