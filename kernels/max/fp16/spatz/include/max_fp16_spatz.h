#ifndef MAX_FP16_SPATZ_H_
#define MAX_FP16_SPATZ_H_

#include <stdint.h>

/* C[k] = max(A[k], B[k]) for `size` elements. */
void MAGIA_max_fp16_spatz(const float16 *A, const float16 *B, float16 *C, uint32_t size);
/* Same operation on this tile only, with A, B, C already in this tile's L1 (C may alias A or B). */
void MAGIA_max_fp16_spatz_l1(const float16 *A, const float16 *B, float16 *C, uint32_t size);

#endif /* MAX_FP16_SPATZ_H_ */
