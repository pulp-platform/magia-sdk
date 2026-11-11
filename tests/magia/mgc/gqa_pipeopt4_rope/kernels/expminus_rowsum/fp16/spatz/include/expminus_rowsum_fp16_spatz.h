#ifndef EXPMINUS_ROWSUM_FP16_SPATZ_H_
#define EXPMINUS_ROWSUM_FP16_SPATZ_H_

#include <stdint.h>

#include "expminus_rowsum_fp16_spatz_params.h"

/* Fixed fields of the parameter block for the scale s: rows, cols and the exp constants. */
static inline void expminus_rowsum_fp16_spatz_set_scale(expminus_rowsum_fp16_spatz_params_t *p,
                                                        float16 s,
                                                        uint32_t rows,
                                                        uint32_t cols)
{
    union { float16 f[2]; uint32_t u; } min_max, coef;

    min_max.f[0] = (float16)(EXPMINUS_ROWSUM_EXP_MIN / (float)s);
    min_max.f[1] = (float16)(EXPMINUS_ROWSUM_EXP_MAX / (float)s);
    coef.f[0]    = (float16)(EXPMINUS_ROWSUM_EXP_COEF * (float)s);
    coef.f[1]    = 0;

    p->dims    = rows | (cols << 16);
    p->min_max = min_max.u;
    p->coef    = coef.u;
}

/* Y[r, c] = exp(s * (X[r, c] - M[r])) and S[r] = sum_c Y[r, c]; if A is not NULL, also
   A[r] = exp(s * (A[r] - M[r])) in place. All operands already in this tile's L1. */
void MAGIA_expminus_rowsum_fp16_spatz_l1(const float16 *X,
                                         const float16 *M,
                                         float16 *Y,
                                         float16 *S,
                                         float16 *A,
                                         float16 s,
                                         uint32_t rows,
                                         uint32_t cols);

#endif /* EXPMINUS_ROWSUM_FP16_SPATZ_H_ */
