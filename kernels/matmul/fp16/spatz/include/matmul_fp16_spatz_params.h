#ifndef MATMUL_FP16_SPATZ_PARAMS_H_
#define MATMUL_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_A;      /* Tile's input tensor shard A      */
    uintptr_t shard_B;      /* Tile's input tensor shard B      */
    uintptr_t shard_Y;      /* Tile's output tensor shard       */

    uint32_t M;             /* Rows of matrix A                 */
    uint32_t K;             /* Columns of A / Rows of B         */
    uint32_t O;             /* Columns of matrix B              */

    uint32_t batch_start;   /* Tile's batch global index        */
    uint32_t batch_len;     /* Number of batches for this tile  */
} matmul_fp16_spatz_params_t;

#endif  /* MATMUL_FP16_SPATZ_PARAMS_H_ */
