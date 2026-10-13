#ifndef COL2IM_FP16_SPATZ_PARAMS_H_
#define COL2IM_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X;       /* Tile's input columns shard               */
    uintptr_t shard_Y;       /* Tile's output image shard                */

    uint32_t batch;          /* Batch size (N)                           */
    uint32_t total_channels; /* Total channels in the global tensor (C)   */

    uint32_t c_start;        /* Tile's channel shard start               */
    uint32_t c_len;          /* Tile's number of channels to process     */

    uint32_t image_h;        /* Reconstructed image height               */
    uint32_t image_w;        /* Reconstructed image width                */

    uint32_t block_h;        /* Block/Kernel height                      */
    uint32_t block_w;        /* Block/Kernel width                       */

    uint32_t pad_h;          /* Vertical padding                         */
    uint32_t pad_w;          /* Horizontal padding                       */

    uint32_t stride_h;       /* Vertical stride                          */
    uint32_t stride_w;       /* Horizontal stride                        */

    uint32_t dilation_h;     /* Vertical dilation                        */
    uint32_t dilation_w;     /* Horizontal dilation                      */

    uint32_t l_len;          /* Total number of columns (L)              */
} col2im_fp16_spatz_params_t;

#endif  /* COL2IM_FP16_SPATZ_PARAMS_H_ */
