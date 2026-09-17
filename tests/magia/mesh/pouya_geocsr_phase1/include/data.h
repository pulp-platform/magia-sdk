#ifndef DATA_H_
#define DATA_H_

#include <stdint.h>

#define DIM_M 10
#define DIM_K 10
#define NNZ   40
#define RANDOM_SEED 12345

/* NOTE: row_ptr below was re-derived from the actual col_idx/values entries
   present in the original data file (they summed to 40 nonzeros, not 48 as
   the original row_ptr/NNZ implied). Fixed here so parser + data agree. */
static const uint32_t row_ptr[] = {0, 3, 8, 9, 15, 21, 26, 30, 34, 39, 40};

static const uint32_t col_idx[] = {
    0, 6, 8,
    0, 5, 6, 8, 9,
    0,
    0, 4, 5, 6, 7, 9,
    1, 4, 5, 7, 8, 9,
    0, 2, 3, 7, 9,
    3, 4, 8, 9,
    0, 2, 6, 7,
    0, 1, 3, 7, 9,
    4
};

/* CSR values, quantized to int16 */
static const int16_t values[] = {
    -4, 3, 9,
    -1, -8, -3, 6, -7,
    2,
    -1, -3, -3, 3, -9, -4,
    -6, -4, -5, 0, 7, 7,
    5, 5, -8, -5, 8,
    -10, 7, 9, -2,
    -5, 1, -4, 0,
    8, 5, -3, -2, 0,
    3
};

static const int16_t x[] = {
    -9, -4, 7, 2, -3, -9, 5, -10, -8, 10
};

/* Expected result, int32 (values x x, accumulated in int32) */
static const int32_t G[] = {
    -21, -52, -18, 110, 95, 104, -133, 32, -78, -9
};

#endif /* DATA_H_ */