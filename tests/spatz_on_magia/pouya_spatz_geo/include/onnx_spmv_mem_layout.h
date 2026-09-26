#ifndef SPMV_SPATZ_MEM_LAYOUT_H
#define SPMV_SPATZ_MEM_LAYOUT_H

#include <stdint.h>
#include "onnx_spmv_params.h"

#define ALIGN_4B(a) (((a) + 3u) & ~3u)

/*
====================================================================
SPATZ chunk size

VLMAX at SEW=32, LMUL=SPATZ_LMUL is (VLEN / 32) * SPATZ_LMUL.
Default HW config is VLEN=256 bit [README], so with LMUL=1:

    VLMAX = 256 / 32 = 8 elements

This is the SAFE / conservative default: e32,m1 keeps the task to a
single vector register per operand (no register-group bookkeeping),
and only vle32/vmv/vmul/vredsum-class instructions are used, none of
which are on the guide's "verified working" list -- they must be
confirmed in simulation before trusting this on real HW [ASSUME].

Once that is confirmed, raising SPATZ_LMUL (2/4/8, if VLEN/HW
supports it) increases the chunk size and reduces the number of
SPATZ calls per row -- pure tuning knob, no other code changes
needed.
====================================================================
*/
#define SPATZ_VLEN_BITS   256
#define SPATZ_LMUL        1
#define SPATZ_MAX_VL      ((SPATZ_VLEN_BITS / 32) * SPATZ_LMUL)

/*
--------------------------------------------------------------------
Per-tile L1 region for the SPATZ params block + the two input chunk
buffers + the scalar result. Chained onto whatever base address the
caller passes in (in main.c: right after this tile's existing SpMV
buffers -- local_x, local_rowptr, valcol ping-pong, local_y), the
same "chain of offsets" style already used for the rest of this
tile's L1 map.

This region is allocated ONCE per tile (not per tile-of-rows, not
per row): every row's chunks reuse it sequentially, since the
control core always waits for SPATZ's result before reusing it.
--------------------------------------------------------------------
*/
typedef struct {
    uintptr_t params;   /* spmv_spatz_params_t                 */
    uintptr_t values;   /* int32_t[SPATZ_MAX_VL]                */
    uintptr_t xvec;      /* int32_t[SPATZ_MAX_VL]                */
    uintptr_t result;    /* int32_t                              */
    uint32_t  end;        /* one past the last byte used          */
} spatz_l1_layout_t;

static inline spatz_l1_layout_t spatz_layout(uint32_t base_addr)
{
    spatz_l1_layout_t L;

    L.params = ALIGN_4B(base_addr);
    L.values = ALIGN_4B(L.params + sizeof(spmv_spatz_params_t));
    L.xvec   = ALIGN_4B(L.values + SPATZ_MAX_VL * sizeof(int32_t));
    L.result = ALIGN_4B(L.xvec   + SPATZ_MAX_VL * sizeof(int32_t));
    L.end    = ALIGN_4B(L.result + sizeof(int32_t));

    return L;
}

/*
Once the real per-tile L1 size (L1_TILE_OFFSET / L1 SPM size) is
known [TODO in guide], add here:

    _Static_assert(spatz_layout(WORST_CASE_BASE).end - L1_BASE_TILE
                    <= L1_SIZE, "SPATZ region overflows tile L1");
*/

#endif /* SPMV_SPATZ_MEM_LAYOUT_H */