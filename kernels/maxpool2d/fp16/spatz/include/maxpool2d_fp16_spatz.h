#ifndef MAXPOOL2D_FP16_SPATZ_H_
#define MAXPOOL2D_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_maxpool2d_fp16_spatz(const float16* X, float16 *Y, uint32_t kernel_h, uint32_t kernel_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w, uint32_t input_shape[4], uint32_t output_shape[4]);

#endif /* MAXPOOL2D_FP16_SPATZ_H_ */
