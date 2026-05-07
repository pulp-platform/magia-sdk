#ifndef TEST_H
#define TEST_H

#include <stdint.h>

/*
Matrix dimensions:
A = 64 x 64
x = 64 x 1
y = 64 x 1
*/

#define M 64
#define N 64

#define MAX_NNZ (M * N)

/*
Dense input matrix (row-major)
Pattern: Row-index is used for non-zero values to keep it unique but predictable.
Even rows have values at even columns, Odd rows have values at odd columns.
*/
static uint16_t A_dense[M * N] = {
    #define R(i) 0, 0, 0, i, 0, 0, 0, 0, 0, i, 0, 0, 0, 0, 0, i, 0, i, 0, 0, 0, i, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, i, 0, i, 0, i, 0, 0, 0, i, 0, i, 0, 0, 0, 0, 0, i, 0, i
    #define S(i) i, 0, i, 0, 0, 0, i, 0, 0, 0, 0, 0, 0, 0, 0, 0, i, 0, 0, 0, i, 0, 0, 0, i, 0, i, 0, 0, 0, 0, 0, i, 0, i, 0, 0, 0, i, 0, 0, 0, i, 0, 0, 0, 0, 0, i, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    S(1), R(2), S(3), R(4), S(5), R(6), S(7), R(8),
    S(9), R(10), S(11), R(12), S(13), R(14), S(15), R(16),
    S(17), R(18), S(19), R(20), S(21), R(22), S(23), R(24),
    S(25), R(26), S(27), R(28), S(29), R(30), S(31), R(32),
    S(33), R(34), S(35), R(36), S(37), R(38), S(39), R(40),
    S(41), R(42), S(43), R(44), S(45), R(46), S(47), R(48),
    S(49), R(50), S(51), R(52), S(53), R(54), S(55), R(56),
    S(57), R(58), S(59), R(60), S(61), R(62), S(63), R(64)
    #undef R
    #undef S
};

/*
Input vector x: alternating 1 and 2
*/
static uint16_t x[N] = {
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2
};

/*
Output vector
*/
static uint16_t y[M] = {0};

/*
Correct Expected Result:
Even rows (0-based): 32 * (i+1)
Odd rows (0-based): 64 * (i+1)
*/
static uint16_t y_expected[M] = {
    12,   48,   36,   96,   60,  144,   84,  192,  108,  240,  132,  288,  156,  336,
    180,  384,  204,  432,  228,  480,  252,  528,  276,  576,  300,  624,  324,  672,
   	348,	720,	372,	768,	396,	816,	420,	864,	444,	912,	468,	960,	492,	1008,
   	516,	1056,	540,	1104,	564,	1152,	588,	1200,	612,	1248,	636,	1296,	660,	1344,
   	684,	1392,	708,	1440,	732,	1488,	756,	1536
};

#endif