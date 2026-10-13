#ifndef ELU_FP16_PARAMS_H_
#define ELU_FP16_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X;      /* Tile's first operand shard       */
    uintptr_t shard_Y;      /* Tile's computed result shard     */
    uintptr_t alpha;        /* Coefficient for ELU              */
    uint32_t start;         /* Tile's shard global start index  */
    uint32_t end;           /* Tile's shard global end index    */
    uint32_t len;           /* Tile's shard len                 */
} elu_fp16_spatz_params_t;

#endif  /* ELU_FP16_PARAMS_H_ */
