#ifndef RESIZE_FP16_SPATZ_PARAMS_H_
#define RESIZE_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X;          /* Tile's input tensor shard          */
    uintptr_t shard_Y;          /* Tile's output tensor shard         */

    uint32_t in_h;              /* Input tensor height dimension      */
    uint32_t in_w;              /* Input tensor width dimension       */
    uint32_t out_h;             /* Output tensor height dimension     */
    uint32_t out_w;             /* Output tensor width dimension      */

    uint32_t iteration_start;   /* Tile's shard global start index    */
    uint32_t iteration_len;     /* Tile's shard length (channels)     */
} resize_fp16_spatz_params_t;

#endif  /* RESIZE_FP16_SPATZ_PARAMS_H_ */
