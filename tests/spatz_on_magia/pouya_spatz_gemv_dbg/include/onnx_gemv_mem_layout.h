#ifndef ONNX_GEMV_H_
#define ONNX_GEMV_H_

#include "tile.h"
#include "data.h"
#include "onnx_gemv_params.h"

#define ALIGNMENT             (4)
/* Aligns the given address to 4-bytes */
#define ALIGN_4B(addr)        (((addr) + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1))

#define L1_BASE_TILE          (L1_BASE + (get_hartid() * L1_TILE_OFFSET))
#define M_K_SIZE              ((DIM_M) * (DIM_K) * sizeof(float16))
#define K_SIZE                ((DIM_K) * sizeof(float16))
#define M_SIZE                ((DIM_M) * sizeof(float16))
#define SCALAR_SIZE           (sizeof(float16))

#define ONNX_GEMV_PARAMS_BASE (L1_BASE_TILE)
#define ONNX_GEMV_PARAMS_SIZE ALIGN_4B(sizeof(onnx_gemv_params_t))

#define A_BASE                ALIGN_4B(ONNX_GEMV_PARAMS_BASE + ONNX_GEMV_PARAMS_SIZE)
#define A_SIZE                ALIGN_4B(M_K_SIZE)

#define X_BASE                ALIGN_4B(A_BASE + A_SIZE)
#define X_SIZE                ALIGN_4B(K_SIZE)

#define C_BASE                ALIGN_4B(X_BASE + X_SIZE)
#define C_SIZE                ALIGN_4B(M_SIZE)

#define Y_BASE                ALIGN_4B(C_BASE + C_SIZE)
#define Y_SIZE                ALIGN_4B(M_SIZE)

#define G_BASE                ALIGN_4B(Y_BASE + Y_SIZE)
#define G_SIZE                ALIGN_4B(M_SIZE)

#define ALPHA_BASE            ALIGN_4B(G_BASE + G_SIZE)
#define ALPHA_SIZE            ALIGN_4B(SCALAR_SIZE)

#define BETA_BASE             ALIGN_4B(ALPHA_BASE + ALPHA_SIZE)
#define BETA_SIZE             ALIGN_4B(SCALAR_SIZE)

#endif /* ONNX_GEMV_H_ */