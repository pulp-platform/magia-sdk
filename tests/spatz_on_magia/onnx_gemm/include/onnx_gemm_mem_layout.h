#ifndef ONNX_GEMM_H_
#define ONNX_GEMM_H_

#include "tile.h"
#include "data.h"
#include "onnx_gemm_params.h"

#define ALIGNMENT   (4)
/* Aligns the given address to 4-bytes */
#define ALIGN_4B(addr)  (((addr) + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1))

#define L1_BASE_TILE    (L1_BASE + (get_hartid() * L1_TILE_OFFSET))
#define M_K_SIZE        ((DIM_M) * (DIM_K) * sizeof(float16))
#define K_N_SIZE        ((DIM_K) * (DIM_N) * sizeof(float16))
#define M_N_SIZE        ((DIM_M) * (DIM_N) * sizeof(float16))
#define SCALAR_SIZE     (sizeof(float16))

#define ONNX_GEMM_PARAMS_BASE   (L1_BASE_TILE)
#define ONNX_GEMM_PARAMS_SIZE   ALIGN_4B(sizeof(onnx_gemm_params_t))

#define A_BASE      ALIGN_4B(ONNX_GEMM_PARAMS_BASE + ONNX_GEMM_PARAMS_SIZE)
#define A_SIZE      ALIGN_4B(M_K_SIZE)

#define B_BASE      ALIGN_4B(A_BASE + A_SIZE)
#define B_SIZE      ALIGN_4B(K_N_SIZE)

#define C_BASE      ALIGN_4B(B_BASE + B_SIZE)
#define C_SIZE      ALIGN_4B(M_N_SIZE)

#define Y_BASE      ALIGN_4B(C_BASE + C_SIZE)
#define Y_SIZE      ALIGN_4B(M_N_SIZE)

#define G_BASE      ALIGN_4B(Y_BASE + Y_SIZE)
#define G_SIZE      ALIGN_4B(M_N_SIZE)

#define ALPHA_BASE  ALIGN_4B(G_BASE + G_SIZE)
#define ALPHA_SIZE  ALIGN_4B(SCALAR_SIZE)

#define BETA_BASE   ALIGN_4B(ALPHA_BASE + ALPHA_SIZE)
#define BETA_SIZE   ALIGN_4B(SCALAR_SIZE)

#endif  /* ONNX_GEMM_H_ */
