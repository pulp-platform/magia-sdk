#ifndef AVERAGEPOOL2D_FP16_SPATZ_PARAMS_H_
#define AVERAGEPOOL2D_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X;       /* Tile's input tensor shard   */
    uintptr_t shard_Y;       /* Tile's output tensor shard  */

    uint32_t c_start;        /* Tile's channel shard start  */
    uint32_t c_len;          /* Tile's number of channels   */

    uint32_t h_in;           /* Input Height (H_in)         */
    uint32_t w_in;           /* Input Width (W_in)          */
    uint32_t h_out;          /* Output Height (H_out)       */
    uint32_t w_out;          /* Output Width (W_out)        */

    uint32_t kernel_h;       /* Kernel Shape Height         */
    uint32_t kernel_w;       /* Kernel Shape Width          */
    uint32_t stride_h;       /* Stride Along Height         */
    uint32_t stride_w;       /* Stride Along Width          */
    uint32_t pad_h;          /* Padding Top/Bottom          */
    uint32_t pad_w;          /* Padding Left/Right          */
} averagepool2d_fp16_spatz_params_t;

#endif  /* AVERAGEPOOL2D_FP16_SPATZ_PARAMS_H_ */
