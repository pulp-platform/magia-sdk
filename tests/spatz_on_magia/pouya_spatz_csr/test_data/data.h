#ifndef DATA_H_
#define DATA_H_

#include <stdint.h>

#define DIM_M 8
#define DIM_K 8
#define NNZ   22   /* padded: each row rounded up to an even nonzero-count so
                     * every row segment starts 4-byte aligned for vle16.v */


static const uint32_t row_ptr[] = {0, 2, 5, 8, 11, 14, 17, 20, 22};

static const uint16_t col_idx[] = {
    0, 1, 
    0, 1, 2, 
    1, 2, 3, 
    2, 3, 4,
    3, 4, 5, 
    4, 5, 6, 
    5, 6, 7, 
    6, 7};

/* CSR nonzero values (NNZ entries) */
static const float16 values[] = {
     2.0f, 3.0f,
     1.0f, 4.0f, 2.0f,
     3.0f, 1.0f, 5.0f, 
     2.0f, 2.0f, 3.0f, 
     1.0f, 3.0f, 2.0f, 
     4.0f, 1.0f, 2.0f, 
     2.0f, 3.0f, 1.0f,
     3.0f, 2.0f};

/* x[k] = k + 1 */
static const float16 x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
/*Golden result*/
static const float16 G[] = {
    8.0f, 15.0f, 29.0f, 29.0f, 31.0f, 40.0f, 41.0f, 37.0f};

#endif /* DATA_H_ */