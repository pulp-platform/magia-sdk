#ifndef ONNX_BATCHNORM_H_
#define ONNX_BATCHNORM_H_

#include "data.h"
#include "tile.h"
#include "onnx_batchnorm_params.h"

#define ALIGNMENT   (4)
/* Aligns the given address to 4-bytes */
#define ALIGN_4B(addr)  (((addr) + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1))

#define L1_BASE_TILE    (L1_BASE + (get_hartid() * L1_TILE_OFFSET))
#define VEC_SIZE        ((LEN) * sizeof(float16))
#define SCALAR_SIZE     (sizeof(float16))

#define ONNX_BATCHNORM_PARAMS_BASE    (L1_BASE_TILE)
#define ONNX_BATCHNORM_PARAMS_SIZE    ALIGN_4B(sizeof(onnx_batchnorm_params_t))

#define GAMMA_BASE  ALIGN_4B(ONNX_BATCHNORM_PARAMS_BASE + ONNX_BATCHNORM_PARAMS_SIZE)
#define GAMMA_SIZE  ALIGN_4B(VEC_SIZE)

#define BETA_BASE   ALIGN_4B(GAMMA_BASE + GAMMA_SIZE)
#define BETA_SIZE   ALIGN_4B(VEC_SIZE)

#define MEAN_BASE   ALIGN_4B(BETA_BASE + BETA_SIZE)
#define MEAN_SIZE   ALIGN_4B(VEC_SIZE)

#define VAR_BASE    ALIGN_4B(MEAN_BASE + MEAN_SIZE)
#define VAR_SIZE    ALIGN_4B(VEC_SIZE)

#define INPUT_BASE  ALIGN_4B(VAR_BASE + VAR_SIZE)
#define INPUT_SIZE  ALIGN_4B(VEC_SIZE)

#define RES_BASE    ALIGN_4B(INPUT_BASE + INPUT_SIZE)
#define RES_SIZE    ALIGN_4B(VEC_SIZE)

#define EXP_BASE    ALIGN_4B(RES_BASE + RES_SIZE)
#define EXP_SIZE    ALIGN_4B(VEC_SIZE)

#define EPS_BASE    ALIGN_4B(EXP_BASE + EXP_SIZE)
#define EPS_SIZE    ALIGN_4B(SCALAR_SIZE)

#endif  /* ONNX_BATCHNORM_H_ */
