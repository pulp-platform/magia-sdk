#ifndef EXPMINUS_ROWSUM_FP16_SPATZ_PARAMS_H_
#define EXPMINUS_ROWSUM_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X; /* Input matrix [rows x cols]                       */
    uintptr_t shard_M; /* Row max [rows]                                   */
    uintptr_t shard_Y; /* Output matrix [rows x cols], may alias X         */
    uintptr_t shard_S; /* Output row sum of Y [rows]                       */
    uintptr_t shard_A; /* Previous row max [rows], in-place, 0 if none     */

    uint32_t dims;    /* Number of rows | row length << 16                     */
    uint32_t min_max; /* fp16 bits of the exp clamps MIN / s | MAX / s << 16   */
    uint32_t coef;    /* fp16 bits of the exp coefficient COEF * s             */
} expminus_rowsum_fp16_spatz_params_t;

/* Fast exp (Schraudolph 1999): exp(x) ~ reinterpret_fp16(COEF * x + BIAS), with x clamped to
   [MIN, MAX]. COEF = 2^mantissa / ln(2), BIAS = exponent_bias * 2^mantissa (FP16). */
#define EXPMINUS_ROWSUM_EXP_MIN  (-10.0f)
#define EXPMINUS_ROWSUM_EXP_MAX  (+10.0f)
#define EXPMINUS_ROWSUM_EXP_COEF (1477.0f)
#define EXPMINUS_ROWSUM_EXP_BIAS (15360.0f)

#endif /* EXPMINUS_ROWSUM_FP16_SPATZ_PARAMS_H_ */
