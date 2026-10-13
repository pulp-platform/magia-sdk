#ifndef CONV2DGEMM_FP16_SPATZ_PARAMS_H_
#define CONV2DGEMM_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_A;       /* Weight slice        [M, K]                      */
    uintptr_t shard_B;       /* im2col matrix       [K, N]                      */
    uintptr_t shard_C;       /* Bias broadcast      [M, N]                      */
    uintptr_t shard_Y;       /* Output slice        [M, N]                      */
    uintptr_t alpha;         /* GEMM A @ B scalar multiplier                    */
    uintptr_t beta;          /* GEMM C scalar multiplier (0 when no bias)       */

    uint32_t M;              /* This tile's output channels (C_out slice)       */
    uint32_t N;              /* H_out * W_out                                   */
    uint32_t K;              /* C_in * K_h * K_w                                */

    uint32_t has_bias;       /* Wether Bias is present or not                   */
    uint32_t n_batches;      /* Number of batches (N)                           */
    uint32_t c_out;          /* Total number of output channels                 */
    uint32_t group;          /* Convolution groups (1 = standard convolution)   */
    uint32_t oc_start;       /* Global index of this tile's first output channel*/
    uint32_t c_in;           /* Input channels (C_in)                           */

    uint32_t h_in;           /* Input Height (H_in)                             */
    uint32_t w_in;           /* Input Width (W_in)                              */
    uint32_t h_out;          /* Output Height (H_out)                           */
    uint32_t w_out;          /* Output Width (W_out)                            */

    uint32_t kernel_h;       /* Kernel Shape Height                             */
    uint32_t kernel_w;       /* Kernel Shape Width                              */
    uint32_t stride_h;       /* Stride Along Height                             */
    uint32_t stride_w;       /* Stride Along Width                              */
    uint32_t pad_h;          /* Padding Top/Bottom                              */
    uint32_t pad_w;          /* Padding Left/Right                              */
} conv2dgemm_fp16_spatz_params_t;

#endif  /* CONV2DGEMM_FP16_SPATZ_PARAMS_H_ */
