#ifndef ONNX_SPMV_H_
#define ONNX_SPMV_H_

#include "tile.h"
#include "data.h"
#include "onnx_spmv_params.h"

#define ALIGNMENT              (4)
#define ALIGN_4B(addr)         (((addr) + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1))

#define L1_BASE_TILE           (L1_BASE + (get_hartid() * L1_TILE_OFFSET))

#define ROW_PTR_SIZE           ((DIM_M + 1) * sizeof(uint32_t))
#define COL_IDX_SIZE           ((NNZ) * sizeof(uint32_t))
#define VALUES_SIZE            ((NNZ) * sizeof(float16))
#define X_SIZE                 ((DIM_K) * sizeof(float16))
#define M_SIZE                 ((DIM_M) * sizeof(float16))

#define ONNX_SPMV_PARAMS_BASE  (L1_BASE_TILE)
#define ONNX_SPMV_PARAMS_SIZE  ALIGN_4B(sizeof(onnx_spmv_params_t))

#define ROW_PTR_BASE           ALIGN_4B(ONNX_SPMV_PARAMS_BASE + ONNX_SPMV_PARAMS_SIZE)
#define ROW_PTR_BASE_SIZE      ALIGN_4B(ROW_PTR_SIZE)

#define COL_IDX_BASE           ALIGN_4B(ROW_PTR_BASE + ROW_PTR_BASE_SIZE)
#define COL_IDX_BASE_SIZE      ALIGN_4B(COL_IDX_SIZE)

#define VALUES_BASE            ALIGN_4B(COL_IDX_BASE + COL_IDX_BASE_SIZE)
#define VALUES_BASE_SIZE       ALIGN_4B(VALUES_SIZE)

#define X_BASE                 ALIGN_4B(VALUES_BASE + VALUES_BASE_SIZE)
#define X_BASE_SIZE            ALIGN_4B(X_SIZE)

#define Y_BASE                 ALIGN_4B(X_BASE + X_BASE_SIZE)
#define Y_BASE_SIZE             ALIGN_4B(M_SIZE)

#define G_BASE                 ALIGN_4B(Y_BASE + Y_BASE_SIZE)

#endif /* ONNX_SPMV_H_ */