#ifndef MUL_BCAST_FP16_SPATZ_PARAMS_H_
#define MUL_BCAST_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

/* B is one row of row_len, shared by every row of A ([1,C,H,W] * [1,1,H,W]). */
#define MUL_BCAST_ROW    (0)
/* B is one scalar per row of A ([1,C,H,W] * [1,C,H,1]).                      */
#define MUL_BCAST_SCALAR (1)
/* One FP32 constant, encoded in constant_bits, for the whole shard. */
#define MUL_BCAST_CONSTANT (2)

typedef struct {
    uintptr_t shard_A;      /* Tile's A shard, [rows, row_len]                */
    uintptr_t shard_B;      /* row_len elements (ROW) or rows elements (SCALAR) */
    uintptr_t shard_Y;      /* Tile's Y shard, [rows, row_len]                */

    uint32_t rows;          /* Rows this tile owns                            */
    uint32_t row_len;       /* Elements per row                               */
    uint32_t mode;          /* MUL_BCAST_*                                    */
    uint32_t constant_bits; /* Used only by MUL_BCAST_CONSTANT                */
} mul_bcast_fp16_spatz_params_t;

#endif  /* MUL_BCAST_FP16_SPATZ_PARAMS_H_ */
