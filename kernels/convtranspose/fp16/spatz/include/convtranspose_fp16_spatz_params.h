#ifndef CONVTRANSPOSE_FP16_SPATZ_PARAMS_H_
#define CONVTRANSPOSE_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X;       /* Tile's input channels shard                     */
    uintptr_t shard_W;       /* Tile's weights shard                            */
    uintptr_t shard_Y;       /* Tile's output channels shard                    */

    uint32_t c_out_start;    /* Tile's output channel shard start               */
    uint32_t c_out_len;      /* Tile's number of output channels                */
    uint32_t c_in_g;         /* Input channels per group (C_in / G)             */
    uint32_t c_out_g;        /* Output channels per group (C_out / G)           */

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
} convtranspose_fp16_spatz_params_t;

#endif  /* CONVTRANSPOSE_FP16_SPATZ_PARAMS_H_ */
