#ifndef ADD_FP16_SPATZ_H_
#define ADD_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_add_fp16_spatz(const float16 *A, const float16 *B, float16 *C, uint32_t size);

void MAGIA_add_bcast_fp16_spatz(const float16 *A, const float16 *B, float16 *Y,
                                uint32_t rows, uint32_t row_len, uint32_t mode);

#endif /* ADD_FP16_SPATZ_H_ */
