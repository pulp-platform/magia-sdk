#ifndef TEST_H
#define TEST_H

#include <stdint.h>

// Matrix size updated to 16x16
#define M 16   
#define N 16   
// We have 16 diagonal elements + 15 super-diagonal elements
#define NNZ 31 

// CSR representation
// values: [row0_val1, row0_val2, row1_val1, row1_val2, ...]
static uint16_t values[NNZ]  = {
    1, 2,  1, 2,  1, 2,  1, 2,  1, 2,  1, 2,  1, 2,  1, 2, 
    1, 2,  1, 2,  1, 2,  1, 2,  1, 2,  1, 2,  1, 2,  1
};

// Column indices corresponding to the values above
static uint16_t col_idx[NNZ] = {
    0, 1,  1, 2,  2, 3,  3, 4,  4, 5,  5, 6,  6, 7,  7, 8,
    8, 9,  9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15
};

// row_ptr: entry i points to the start of row i in values/col_idx
static uint16_t row_ptr[M+1] = {
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 31
};

// Input vector x (1 to 16)
static uint16_t x[N] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
};

// Output
static uint16_t y[M] = {0};

// Expected result (y = Ax)
// For rows 0-14: y[i] = (1 * x[i]) + (2 * x[i+1])
// For row 15:    y[15] = (1 * x[15])
static uint16_t y_expected[M] = {
    5,   // (1*1) + (2*2)
    8,   // (1*2) + (2*3)
    11,  // (1*3) + (2*4)
    14,  // ...
    17,
    20,
    23,
    26,
    29,
    32,
    35,
    38,
    41,
    44,
    47,
    16   // (1*16)
};

#endif