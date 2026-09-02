#ifndef MATMUL_FP16_SPATZ_H_
#define MATMUL_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_matmul_fp16_spatz(const float16 *A, const float16 *B, float16 *Y, uint32_t M, uint32_t K, uint32_t O, uint32_t total_batches, uint32_t a_batched, uint32_t b_batched);

#endif  /* MATMUL_FP16_SPATZ_H_ */
