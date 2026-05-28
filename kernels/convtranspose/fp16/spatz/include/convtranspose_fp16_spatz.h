#ifndef CONVTRANSPOSE_FP16_SPATZ_H_
#define CONVTRANSPOSE_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_convtranspose_fp16_spatz(const float16 *X, const float16 *W, float16 *Y, uint32_t kernel_h, uint32_t kernel_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w, uint32_t num_groups);

#endif  /* CONVTRANSPOSE_FP16_SPATZ_H_ */
