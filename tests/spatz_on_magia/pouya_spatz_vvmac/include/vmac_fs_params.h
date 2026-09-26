#ifndef VMAC_FS_PARAMS_H_
#define VMAC_FS_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t chunk_A; /* First input vector chunk       */
    uintptr_t chunk_B; /* Second input vector chunk      */
    uintptr_t chunk_C; /* Output scalar (dot product)    */
    uint32_t len;      /* Vector length                  */
} vmac_fs_params_t;

#endif /* VMAC_FS_PARAMS_H_ */