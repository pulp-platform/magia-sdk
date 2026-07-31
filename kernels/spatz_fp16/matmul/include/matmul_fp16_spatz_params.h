#ifndef MATMUL_FP16_SPATZ_PARAMS_H_
#define MATMUL_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_A;      /* Tile's rows of A, [a_batches][M][K]           */
    uintptr_t shard_B;      /* B, [b_batches][K][O] - not sharded            */
    uintptr_t shard_Y;      /* Tile's rows of Y, [batch_len][M][O]           */

    uint32_t M;             /* Rows of A this tile owns                      */
    uint32_t K;             /* Columns of A / rows of B                      */
    uint32_t O;             /* Columns of B                                  */

    uint32_t M_total;       /* Rows of A in the whole matrix                 */
    uint32_t m_start;       /* First row of A this tile owns                 */

    uint32_t batch_len;     /* Batches this tile runs (all of them)          */
    uint32_t a_batched;     /* 0 when one A is shared by every batch         */
    uint32_t b_batched;     /* 0 when one B is shared by every batch         */
} matmul_fp16_spatz_params_t;

#endif  /* MATMUL_FP16_SPATZ_PARAMS_H_ */
