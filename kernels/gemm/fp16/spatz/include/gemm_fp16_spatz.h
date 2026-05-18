#ifndef GEMM_FP16_SPATZ_H_
#define GEMM_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_gemm_fp16_spatz(const float16 *A, const float16 *B, const float16 *C, float16 alpha, float16 beta, int transA, int transB, float16 *Y);

#endif  /* GEMM_FP16_SPATZ_H_ */
