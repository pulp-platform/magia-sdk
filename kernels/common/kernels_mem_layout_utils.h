#ifndef KERNELS_MEM_LAYOUT_UTILS_H_
#define KERNELS_MEM_LAYOUT_UTILS_H_

#include "magia_tile_utils.h"
#include "magia_utils.h"

#define ALIGNMENT   (4)

/* Aligns the given address to 4-byte  */
#define ALIGN_4B(addr)  (((addr) + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1))

/* Division with round upwards */
#define DIV_UP(a, b)    (((a) + (b) - 1) / (b))

#endif  /* KERNELS_MEM_LAYOUT_UTILS_H_ */
