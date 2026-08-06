#ifndef GEMM_FP16_SPATZ_PARAMS_H_
#define GEMM_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_A;      /* Tile's input tensor shard A                              */
    uintptr_t shard_B;      /* Tile's input tensor shard B                              */
    uintptr_t shard_C;      /* Tile's input tensor shard C                              */
    uintptr_t shard_Y;      /* Tile's output tensor shard                               */

    uintptr_t alpha;        /* Scalar multiplier for the product of input tensors A * B */
    uintptr_t beta;         /* Scalar multiplier for input tensor C                     */

    int transA;             /* Whether A should be transposed                           */
    int transB;             /* Whether B should be transposed                           */

    int shard_dim;          /* Sharded output dimension: 0 = rows (M), 1 = columns (N)  */
    uint32_t m_start;       /* Tile's shard start along the sharded dimension           */
    uint32_t m_len;         /* Tile's shard length along the sharded dimension          */

    uint32_t M;             /* Rows of A and C (full)                                   */
    uint32_t N;             /* Columns of B and C (full)                                */
    uint32_t K;             /* Columns of A - Rows of B                                 */
} gemm_fp16_spatz_params_t;

#endif  /* GEMM_FP16_SPATZ_PARAMS_H_ */
