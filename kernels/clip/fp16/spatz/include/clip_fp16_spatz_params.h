#ifndef CLIP_FP16_SPATZ_PARAMS_H_
#define CLIP_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_input;  /* Tile's input shard                                       */
    uintptr_t shard_output; /* Tile's computed result shard                             */
    uintptr_t min;          /* Minimum value, under which element is replaced by min    */
    uintptr_t max;          /* Maximum value, above which element is replaced by max    */
    uint32_t start;         /* Tile's shard global start index                          */
    uint32_t end;           /* Tile's shard global end index                            */
    uint32_t len;           /* Tile's shard len                                         */
} clip_fp16_spatz_params_t;

#endif  /* CLIP_FP16_SPATZ_PARAMS_H_ */
