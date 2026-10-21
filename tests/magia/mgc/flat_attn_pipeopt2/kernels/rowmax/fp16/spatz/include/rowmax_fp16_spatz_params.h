#ifndef ROWMAX_FP16_SPATZ_PARAMS_H_
#define ROWMAX_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X; /* Input matrix [rows x cols]                */
    uintptr_t shard_P; /* Previous row max [rows], 0 if there is none */
    uintptr_t shard_M; /* Output row max [rows]                     */

    uint32_t dims; /* Number of rows | row length << 16 */
} rowmax_fp16_spatz_params_t;

#endif /* ROWMAX_FP16_SPATZ_PARAMS_H_ */
