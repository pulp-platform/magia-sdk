#ifndef MEMSET_FP16_SPATZ_H_
#define MEMSET_FP16_SPATZ_H_

#include <stdint.h>

/* C[k] = 0 for `size` elements, with C already in this tile's L1. */
void MAGIA_memset_zero_fp16_spatz_l1(float16 *C, uint32_t size);

#endif /* MEMSET_FP16_SPATZ_H_ */
