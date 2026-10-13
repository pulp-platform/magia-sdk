#ifndef SELU_FP16_PARAMS_H_
#define SELU_FP16_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X;      /* Tile's first operand shard       */
    uintptr_t shard_Y;      /* Tile's computed result shard     */
    uintptr_t alpha;        /* Coefficient for SELU             */
    uintptr_t gamma;        /* Coefficient for SELU             */
    uint32_t start;         /* Tile's shard global start index  */
    uint32_t end;           /* Tile's shard global end index    */
    uint32_t len;           /* Tile's shard len                 */
} selu_fp16_spatz_params_t;

#endif  /* SELU_FP16_PARAMS_H_ */
