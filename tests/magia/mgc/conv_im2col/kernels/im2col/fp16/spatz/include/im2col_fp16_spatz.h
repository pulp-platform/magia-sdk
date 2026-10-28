#ifndef IM2COL_FP16_SPATZ_H_
#define IM2COL_FP16_SPATZ_H_

#include <stdint.h>

#include "im2col_fp16_spatz_params.h"

/* im2col of one block of output pixels on this tile's Spatz (see the task), all operands already in this tile's
   L1; waits for it. */
void MAGIA_im2col_fp16_spatz_l1(const im2col_fp16_spatz_params_t *params);

#endif /* IM2COL_FP16_SPATZ_H_ */
