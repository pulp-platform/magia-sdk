#ifndef SPMV_MAC_PARAMS_H_
#define SPMV_MAC_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t chunk_VAL; /* CSR value chunk for the current row segment, as float */
    uintptr_t chunk_X;   /* Gathered x chunk for the segment, as float            */
    uintptr_t chunk_RES; /* Output: partial dot product of this segment, as float */
    uint32_t len;        /* Segment length (<= SPATZ_MAX_VL)                       */
} spmv_mac_params_t;

#endif /* SPMV_MAC_PARAMS_H_ */