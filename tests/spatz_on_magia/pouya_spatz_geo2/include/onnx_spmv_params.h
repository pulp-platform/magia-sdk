#ifndef SPMV_PARAMS_H_
#define SPMV_PARAMS_H_

#include <stdint.h>

/* Max elements per SPATZ call. e32 + LMUL=8 with VLEN=512 -> 128 elements.
 * Change this if your SPATZ VLEN is different. */
#ifndef SPATZ_MAX_VL
#define SPATZ_MAX_VL 128
#endif

typedef struct {
    uintptr_t addr_val; /* int32 vector: CSR values                      */
    uintptr_t addr_x;   /* int32 vector: matching x elements (sign-ext)  */
    uintptr_t addr_res; /* int32 output: sum(val[i] * x[i])              */
    uint32_t  len;      /* number of valid elements (1..SPATZ_MAX_VL)    */
} spmv_params_t;

#endif /* SPMV_PARAMS_H_ */