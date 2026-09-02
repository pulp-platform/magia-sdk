#ifndef GROUPNORM_FP16_SPATZ_PARAMS_H_
#define GROUPNORM_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X;       /* Tile's input tensor shard                     */
    uintptr_t shard_Y;       /* Tile's output tensor shard                    */

    uintptr_t gamma;         /* GroupNorm scale tensor                        */
    uintptr_t beta;          /* GroupNorm bias tensor (all C)                 */
    uintptr_t eps;           /* GroupNorm epsilon scalar                      */

    uint32_t g_start;        /* Tile's group shard start                      */
    uint32_t g_len;          /* Tile's number of groups                       */
    uint32_t c_per_g;        /* Number of channels per group                  */
    uint32_t hw_len;         /* Elements per channel (H * W)                  */
    uint32_t num_groups;     /* Total number of groups                        */
    uint32_t c_out;          /* Total number of output channels               */

    uint32_t input_len;      /* Total elements of the input tensor            */
    uint32_t output_len;     /* Local elements of the output shard            */
} groupnorm_fp16_spatz_params_t;

#endif  /* GROUPNORM_FP16_SPATZ_PARAMS_H_ */
