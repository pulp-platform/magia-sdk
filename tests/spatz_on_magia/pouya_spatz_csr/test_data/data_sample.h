#ifndef DATA_H_
#define DATA_H_

#include <stdint.h>

#define DIM_M 8
#define DIM_K 8
#define NNZ   19

static const uint32_t row_ptr[] = {0, 3, 5, 6, 8, 9, 11, 14, 19};

/* CSR column indices (NNZ entries). Padding entries (col=0) are harmless since their paired value is 0. */
static const uint32_t col_idx[] = {
    5, 6, 7,      /* row0 */
    2, 4,      /* row1 (+pad) */
    4,      /* row2 (+pad) */
    3, 7,      /* row3 (+pad) */
    0,      /* row4 (+pad) */
    1, 5,     /* row5 (+pad) */
    0, 1, 2,      /* row6 (+pad) */
    1, 2, 4, 6, 7      /* row7 */
};

/* CSR nonzero values (NNZ entries) */
static const float16 values[] = {
    0.243164f, 0.050049f, 6.773438f,      /* row0 */
    -5.851562f, -9.898438f,      /* row1 (+pad) */
    -7.191406f,      /* row2 (+pad) */
    3.550781f, 0.858887f,      /* row3 (+pad) */
    -2.250000f,      /* row4 (+pad) */
    -3.437500f, -8.898438f,      /* row5 (+pad) */
    4.953125f, -6.269531f, 3.583984f,      /* row6 (+pad) */
    -9.617188f, 3.484375f, 0.930664f, -3.583984f, 2.228516f      /* row7 */
};

static const float16 x[] = {
2.933594f, 3.519531f, -2.281250f, 5.328125f, 0.071655f, -4.054688f, 8.234375f, 1.976562f
};

static const float16 G[] = {
12.812500f, 12.640625f, -0.515137f, 20.609375f, -6.601562f, 23.984375f, -15.710938f, -66.812500f
};

#endif /* DATA_H_ */