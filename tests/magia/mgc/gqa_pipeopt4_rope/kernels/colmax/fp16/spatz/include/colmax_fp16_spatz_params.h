#ifndef COLMAX_FP16_SPATZ_PARAMS_H_
#define COLMAX_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X; /* Input matrix [rows x cols] */
    uintptr_t shard_M; /* Output column max [cols]   */

    uint32_t dims; /* Number of rows | row length << 16 */
} colmax_fp16_spatz_params_t;

#endif /* COLMAX_FP16_SPATZ_PARAMS_H_ */
