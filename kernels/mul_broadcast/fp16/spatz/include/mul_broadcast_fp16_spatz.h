#ifndef MUL_BROADCAST_FP16_SPATZ_H_
#define MUL_BROADCAST_FP16_SPATZ_H_

#include <stdint.h>

/* C = A * B[0]: B points to a single scalar, broadcast over all total_elems elements of A. */
void MAGIA_mul_broadcast_fp16_spatz(const float16 *A,
                                    const float16 *B,
                                    float16 *C,
                                    uint32_t total_elems);
/* Same operation on this tile only, with A, B, C already in this tile's L1. */
void MAGIA_mul_broadcast_fp16_spatz_l1(const float16 *A, const float16 *B, float16 *C, uint32_t n);

#endif /* MUL_BROADCAST_FP16_SPATZ_H_ */
