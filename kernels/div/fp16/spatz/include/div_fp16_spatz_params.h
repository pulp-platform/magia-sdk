#ifndef DIV_FP16_PARAMS_H_
#define DIV_FP16_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_A;      /* Tile's first operand shard       */
    uintptr_t shard_B;      /* Tile's second operand shard      */
    uintptr_t shard_C;      /* Tile's computed result shard     */
    uint32_t start;         /* Tile's shard global start index  */
    uint32_t end;           /* Tile's shard global end index    */
    uint32_t len;           /* Tile's shard len                 */
} div_fp16_spatz_params_t;

#endif  /* DIV_FP16_PARAMS_H_ */
