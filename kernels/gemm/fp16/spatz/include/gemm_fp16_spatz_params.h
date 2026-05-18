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

    int transA;            /* Whether A should be transposed                           */
    int transB;            /* Whether A should be transposed                           */

    uint32_t m_start;       /* Tile's A-columns shard start                             */
    uint32_t m_len;         /* Tile's number of elements of A-columns                   */

    uint32_t M;             /* Rows of A anc C                                          */
    uint32_t N;             /* Columns of B and C                                       */
    uint32_t K;             /* Columns of A - Rows of B                                 */
} gemm_fp16_spatz_params_t;

#endif  /* GEMM_FP16_SPATZ_PARAMS_H_ */
