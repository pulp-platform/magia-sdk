#ifndef DATA_H_
#define DATA_H_

#include <stdint.h>

#define DIM_M 8
#define DIM_K 8
#define NNZ   28   /* padded: each row rounded up to an even nonzero-count so
                     * every row segment starts 4-byte aligned for vle16.v */

/* CSR row pointer (DIM_M + 1 entries), padded row lengths: 2,4,4,4,4,4,4,2 */
static const uint32_t row_ptr[] = {0, 2, 6, 10, 14, 18, 22, 26, 28};

/* CSR column indices (NNZ entries). Padding entries (col=0) are harmless
 * since their paired value is 0.0. */
static const uint32_t col_idx[] = {
    0, 1,                /* row0 */
    0, 1, 2, 0,          /* row1 (+pad) */
    1, 2, 3, 0,          /* row2 (+pad) */
    2, 3, 4, 0,          /* row3 (+pad) */
    3, 4, 5, 0,          /* row4 (+pad) */
    4, 5, 6, 0,          /* row5 (+pad) */
    5, 6, 7, 0,          /* row6 (+pad) */
    6, 7};               /* row7 */

/* CSR nonzero values (NNZ entries) */
static const float16 values[] = {
     2.0f, 3.0f,
     1.0f, 4.0f, 2.0f, 0.0f,
     3.0f, 1.0f, 5.0f, 0.0f,
     2.0f, 2.0f, 3.0f, 0.0f,
     1.0f, 3.0f, 2.0f, 0.0f,
     4.0f, 1.0f, 2.0f, 0.0f,
     2.0f, 3.0f, 1.0f, 0.0f,
     3.0f, 2.0f};

/* x[k] = k + 1 */
static const float16 x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

/*
 * Golden model unchanged by padding - zero-valued entries contribute 0 to
 * the row sum. All products (max 40) and row sums (max 41) stay integers
 * well below 2^11 = 2048, exactly representable in fp16.
 */
static const float16 G[] = {
    8.0f, 15.0f, 29.0f, 29.0f, 31.0f, 40.0f, 41.0f, 37.0f};

#endif /* DATA_H_ */