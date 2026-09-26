#ifndef DATA_H_
#define DATA_H_

#include <stdint.h>

#define DIM_M 8
#define DIM_K 8
#define NNZ   22

/* row_ptr fixed to match the actual (unpadded) row lengths below
 * (2,3,3,3,3,3,3,2 -> sum 22). fp32/e32 has no 4-byte-alignment need for
 * "every row starts aligned" the way the fp16 version did, since each
 * element is already 4 bytes -- so no padding is required here. */
static const uint32_t row_ptr[] = {0, 2, 5, 8, 11, 14, 17, 20, 22};

static const uint32_t col_idx[] = {
    0, 1,
    0, 1, 2,
    1, 2, 3,
    2, 3, 4,
    3, 4, 5,
    4, 5, 6,
    5, 6, 7,
    6, 7};

/* CSR nonzero values (NNZ entries) */
static const float values[] = {
     2.0f, 3.0f,
     1.0f, 4.0f, 2.0f,
     3.0f, 1.0f, 5.0f,
     2.0f, 2.0f, 3.0f,
     1.0f, 3.0f, 2.0f,
     4.0f, 1.0f, 2.0f,
     2.0f, 3.0f, 1.0f,
     3.0f, 2.0f};

/* x[k] = k + 1 */
static const float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

/* Golden result */
static const float G[] = {
    8.0f, 15.0f, 29.0f, 29.0f, 31.0f, 40.0f, 41.0f, 37.0f};

#endif /* DATA_H_ */