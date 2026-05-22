#ifndef CLIP_FP16_SPATZ_H_
#define CLIP_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_clip_fp16_spatz(const float16 *input, float16 *output, float16 min, float16 max, uint32_t size);

#endif  /* CLIP_FP16_SPATZ_H_ */
