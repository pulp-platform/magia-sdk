#ifndef DATA_H_
#define DATA_H_

#define VEC_LEN   (32)
#define NUM_TESTS (5)

/* 5 pairs of input vectors. Values are small integers (as float) so the
 * dot product is exact in float32 regardless of the summation order used
 * by the hardware reduction. */
static const float A[NUM_TESTS][VEC_LEN] = {
    { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 1.0f, 2.0f, 3.0f, 4.0f },
    { 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 2.0f, 3.0f, 4.0f, 5.0f },
    { 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 3.0f, 4.0f, 5.0f, 6.0f },
    { 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 4.0f, 5.0f, 6.0f, 7.0f },
    { 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 5.0f, 6.0f, 7.0f, 8.0f }
};

static const float B[NUM_TESTS][VEC_LEN] = {
    { 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f },
    { 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f },
    { 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f },
    { 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f },
    { 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.0f, 3.0f }
};

/* Golden dot product of A[t] . B[t] for each test t */
static const float GOLDEN[NUM_TESTS] = { 468.0f, 593.0f, 718.0f, 843.0f, 968.0f };

#endif /* DATA_H_ */