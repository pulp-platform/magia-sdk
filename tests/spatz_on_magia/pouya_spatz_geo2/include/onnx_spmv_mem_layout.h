#ifndef SPMV_MEM_LAYOUT_H_
#define SPMV_MEM_LAYOUT_H_

#include <stdint.h>
#include "onnx_spmv_params.h"

#define ALIGNMENT      (4)
#define ALIGN_4B(addr) (((addr) + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1))

/*
 * L1 layout of one tile (low -> high address):
 *
 *   x            local_x_count * int16   (MUST stay first: valcol
 *                                         addresses are encoded offline)
 *   rowptr       (local_rows + 1) * uint32
 *   valcol buf0  2 * MAX_TILE_NNZ * uint32
 *   valcol buf1  2 * MAX_TILE_NNZ * uint32
 *   y            local_rows * int32
 *   params       spmv_params_t          (read by SPATZ)
 *   vec_val      SPATZ_MAX_VL * int32   (SPATZ input 1)
 *   vec_x        SPATZ_MAX_VL * int32   (SPATZ input 2)
 *   result       1 * int32              (SPATZ output)
 */
typedef struct {
    uint32_t x;
    uint32_t rowptr;
    uint32_t valcol0;
    uint32_t valcol1;
    uint32_t y;
    uint32_t params;
    uint32_t vec_val;
    uint32_t vec_x;
    uint32_t result;
    uint32_t end;
} spmv_l1_layout_t;

static inline spmv_l1_layout_t spmv_l1_layout(uint32_t l1_base,
                                              uint32_t local_x_count,
                                              uint32_t local_rows,
                                              uint32_t tile_buffer_bytes)
{
    spmv_l1_layout_t L;

    L.x       = l1_base;
    L.rowptr  = L.x + local_x_count * sizeof(int16_t);
    L.rowptr  = ALIGN_4B(L.rowptr);
    L.valcol0 = L.rowptr + (local_rows + 1) * sizeof(uint32_t);
    L.valcol1 = L.valcol0 + tile_buffer_bytes;
    L.y       = L.valcol1 + tile_buffer_bytes;
    L.params  = ALIGN_4B(L.y + local_rows * sizeof(int32_t));
    L.vec_val = ALIGN_4B(L.params + sizeof(spmv_params_t));
    L.vec_x   = ALIGN_4B(L.vec_val + SPATZ_MAX_VL * sizeof(int32_t));
    L.result  = ALIGN_4B(L.vec_x + SPATZ_MAX_VL * sizeof(int32_t));
    L.end     = ALIGN_4B(L.result + sizeof(int32_t));

    return L;
}

#endif /* SPMV_MEM_LAYOUT_H_ */