#ifndef COL2IM_FP16_SPATZ_H_
#define COL2IM_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_col2im_fp16_spatz(const float16 *input, float16 *output, uint32_t batch, uint32_t channels, uint32_t image_h, uint32_t image_w, uint32_t block_h, uint32_t block_w, uint32_t pad_h, uint32_t pad_w, uint32_t stride_h, uint32_t stride_w, uint32_t dilation_h, uint32_t dilation_w, uint32_t l_len);

#endif  /* COL2IM_FP16_SPATZ_H_ */
