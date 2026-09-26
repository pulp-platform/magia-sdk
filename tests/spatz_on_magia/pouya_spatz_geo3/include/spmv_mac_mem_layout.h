#ifndef SPMV_MAC_H_
#define SPMV_MAC_H_

#include "spmv_mac_params.h"

#define ALIGNMENT      (4)
#define ALIGN_4B(addr) (((addr) + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1))

/* Max number of elements SPATZ can be handed in a single call. A row
 * longer than this must be split into several calls, accumulated by the
 * control core. */
#define SPATZ_MAX_VL (512)

/* Scratch vectors hold float (see spmv_mac_task.c for why); happens to
 * be the same 4 bytes/element as int32_t on this target either way. */
#define SPMV_MAC_PARAMS_SIZE ALIGN_4B(sizeof(spmv_mac_params_t))
#define SPMV_MAC_VAL_SIZE    ALIGN_4B(SPATZ_MAX_VL * sizeof(float))
#define SPMV_MAC_X_SIZE      ALIGN_4B(SPATZ_MAX_VL * sizeof(float))
#define SPMV_MAC_RES_SIZE    ALIGN_4B(sizeof(float))

#endif /* SPMV_MAC_H_ */