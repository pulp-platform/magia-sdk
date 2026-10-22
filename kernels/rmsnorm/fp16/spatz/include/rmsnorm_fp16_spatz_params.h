#ifndef RMSNORM_FP16_SPATZ_PARAMS_H_
#define RMSNORM_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X; /* Tile's input rows                     */
    uintptr_t shard_Y; /* Tile's output rows (may alias shard_X) */
    uintptr_t gamma;   /* RMSNorm scale, w_len elements         */
    uintptr_t eps;     /* RMSNorm epsilon (fp16 scalar)         */

    uint32_t r_start; /* Tile's row shard start                */
    uint32_t r_len;   /* Tile's number of rows                 */
    uint32_t w_len;   /* Row length (size of normalized axis)  */
} rmsnorm_fp16_spatz_params_t;

#endif /* RMSNORM_FP16_SPATZ_PARAMS_H_ */
