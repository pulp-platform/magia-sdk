#ifndef ROWSLICE_SUM_DIV_FP16_SPATZ_PARAMS_H_
#define ROWSLICE_SUM_DIV_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_Y; /* Output slice [rows x cols]                           */
    uintptr_t shard_O; /* Own slice [rows x cols]                              */
    uintptr_t shard_G; /* Slices of the other tiles [n x rows x cols], own unused */
    uintptr_t shard_S; /* Row divisors [rows]                                  */

    uint32_t dims;  /* Number of rows | row length << 16 */
    uint32_t slices; /* n | own slice index << 16 */
} rowslice_sum_div_fp16_spatz_params_t;

#endif /* ROWSLICE_SUM_DIV_FP16_SPATZ_PARAMS_H_ */
