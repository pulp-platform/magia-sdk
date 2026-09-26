#ifndef SPMV_SPATZ_PARAMS_H
#define SPMV_SPATZ_PARAMS_H

#include <stdint.h>

/*
====================================================================
Parameter block shared between the CV32 host and the SPATZ task.

Only 32-bit fields (uintptr_t / uint32_t) so host and task agree on
the same in-memory layout [per guide sec. 6].

The control core (host) has already, for this call:
  - picked out up to SPATZ_MAX_VL nonzeros of the current row,
  - copied their values into a dense int32 array,
  - gathered the matching x elements (sign-extended to int32) into
    a second dense array, in the same order.

SPATZ therefore does not gather/index anything itself -- it just
multiplies two same-length dense vectors element-wise and reduces.
All data handling (chunking, gathering, partial-row accumulation)
is done by the control core, as requested.
====================================================================
*/
typedef struct {
    uintptr_t addr_values;   /* L1 addr of dense int32 values[len]      */
    uintptr_t addr_xvec;     /* L1 addr of dense int32 x[len] (gathered)*/
    uintptr_t addr_result;   /* L1 addr where task writes int32 result  */
    uint32_t  len;           /* number of elements in this chunk        */
} spmv_spatz_params_t;

#endif /* SPMV_SPATZ_PARAMS_H */