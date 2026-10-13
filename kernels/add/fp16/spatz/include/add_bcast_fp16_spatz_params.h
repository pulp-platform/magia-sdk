#ifndef ADD_BCAST_FP16_SPATZ_PARAMS_H_
#define ADD_BCAST_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

/* B is one row of row_len, shared by every row of A (a bias over the last axis). */
#define ADD_BCAST_ROW    (0)
/* B is one scalar per row of A.                                                  */
#define ADD_BCAST_SCALAR (1)

typedef struct {
    uintptr_t shard_A;      /* Tile's A shard, [rows, row_len]                  */
    uintptr_t shard_B;      /* row_len elements (ROW) or rows elements (SCALAR) */
    uintptr_t shard_Y;      /* Tile's Y shard, [rows, row_len]                  */

    uint32_t rows;          /* Rows this tile owns                              */
    uint32_t row_len;       /* Elements per row                                 */
    uint32_t mode;          /* ADD_BCAST_*                                      */
} add_bcast_fp16_spatz_params_t;

#endif  /* ADD_BCAST_FP16_SPATZ_PARAMS_H_ */
