#ifndef VMAC_FS_H_
#define VMAC_FS_H_

#include "data.h"
#include "magia_tile_utils.h"
#include "magia_utils.h"
#include "vmac_fs_params.h"

#define ALIGNMENT       (4)

/* Aligns the given address to 4-byte  */
#define ALIGN_4B(addr)  (((addr) + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1))

/* Division with round upwards */
#define DIV_UP(a, b)    (((a) + (b) - 1) / (b))

#define L1_BASE_TILE    (L1_BASE + (get_hartid() * L1_TILE_OFFSET))

#define VMAC_PARAMS_BASE (L1_BASE_TILE)
#define VMAC_PARAMS_SIZE ALIGN_4B(sizeof(vmac_fs_params_t))

#define CHUNK_A_BASE    ALIGN_4B(VMAC_PARAMS_BASE + VMAC_PARAMS_SIZE)
#define CHUNK_A_SIZE    ALIGN_4B(VEC_LEN * sizeof(float))

#define CHUNK_B_BASE    ALIGN_4B(CHUNK_A_BASE + CHUNK_A_SIZE)
#define CHUNK_B_SIZE    ALIGN_4B(VEC_LEN * sizeof(float))

#define CHUNK_C_BASE    ALIGN_4B(CHUNK_B_BASE + CHUNK_B_SIZE)
#define CHUNK_C_SIZE    ALIGN_4B(sizeof(float))

/* mmio helper for 32-bit float access, same style as mmio_fp16 */
#define mmio_fp32(addr) (*(volatile float *)(addr))

#endif /* VMAC_FS_H_ */