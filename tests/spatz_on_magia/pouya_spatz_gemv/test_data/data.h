/* Simple hand-crafted header for Spatz ONNX GEMV (8x8 bring-up case) */
#ifndef DATA_H_
#define DATA_H_

#define DIM_M   8
#define DIM_K   8

#define TRANS_A 0

static const float16 ALPHA = 1.000000f;
static const float16 BETA  = 1.000000f;

/* A[m][k] = m * DIM_K + k  (row-major, values 0..63) */
static const float16 A[] = {
     0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
     8.0f,  9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,
    24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f,
    32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f,
    40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f,
    48.0f, 49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f,
    56.0f, 57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f};

/* x[k] = k */
static const float16 x[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

/* C[m] = m  (input vector scaled by beta) */
static const float16 C[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

/*
 * Golden model: G[m] = alpha * sum_k( A[m][k] * x[k] ) + beta * C[m]
 *             = 1 * (8*m*28 + 140) + 1 * m
 *             = 225*m + 140
 *
 * All partial products (max 7*63=441) and partial sums (max 1715) are
 * integers well below 2^11 = 2048, so they are exactly representable in
 * fp16 with zero rounding error - the golden values below will match the
 * hardware output bit-for-bit regardless of vector-reduction order.
 */
static const float16 G[] = {
    140.0f, 365.0f, 590.0f, 815.0f, 1040.0f, 1265.0f, 1490.0f, 1715.0f};

#endif /* DATA_H_ */