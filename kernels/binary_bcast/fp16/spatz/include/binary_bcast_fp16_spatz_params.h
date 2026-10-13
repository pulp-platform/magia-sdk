#ifndef BINARY_BCAST_FP16_SPATZ_PARAMS_H_
#define BINARY_BCAST_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

#define BINARY_BCAST_ROW 0u
#define BINARY_BCAST_SCALAR 1u
#define BINARY_BCAST_SUB 0u
#define BINARY_BCAST_DIV 1u

typedef struct {
    uintptr_t shard_A;
    uintptr_t shard_B;
    uintptr_t shard_Y;
    uint32_t rows;
    uint32_t row_len;
    uint32_t mode;
    uint32_t operation;
} binary_bcast_fp16_spatz_params_t;

#endif
