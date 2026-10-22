#ifndef REDUCEMAX_FP16_SPATZ_H_
#define REDUCEMAX_FP16_SPATZ_H_

#include <stdint.h>

/* Y = max of X over one axis, with X viewed as [outer_dim, reduce_dim, inner_dim] and
   Y as [outer_dim, inner_dim]. */
void MAGIA_reducemax_fp16_spatz(
    const float16 *X, float16 *Y, uint32_t outer_dim, uint32_t reduce_dim, uint32_t inner_dim);
/* Same operation on this tile only, with X, Y already in this tile's L1. */
void MAGIA_reducemax_fp16_spatz_l1(
    const float16 *X, float16 *Y, uint32_t outer_len, uint32_t reduce_dim, uint32_t inner_dim);

#endif /* REDUCEMAX_FP16_SPATZ_H_ */
