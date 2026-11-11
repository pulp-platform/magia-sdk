#ifndef IM2COL_OPT_FP16_SPATZ_PARAMS_H_
#define IM2COL_OPT_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t src; /* Input image rows from lo on (all the rows the block reads), columns
                      src_x .. src_x + src_w - 1, HWC [n_rows x src_w x cin] */
    uintptr_t dst; /* im2col block [cp x (ksize * ksize * cin)], columns in (ky, kx, ci) order    */

    uint32_t height; /* Image rows                                              */
    uint32_t width;  /* Image columns                                           */
    uint32_t cin;    /* Input channels                                          */
    uint32_t ksize;  /* Kernel side (odd)                                       */
    uint32_t stride; /* Stride (rows and columns)                               */
    uint32_t pad;    /* Zero padding on each side, (ksize - 1) / 2              */
    uint32_t row;    /* Output row of the block                                 */
    uint32_t lo;     /* Image row of the first row of src                       */
    uint32_t x0;     /* Output column of the first pixel of the block           */
    uint32_t cp;     /* Pixels of the block (output columns x0 .. x0 + cp - 1)  */
    uint32_t src_w;  /* Columns of a row of src (width: whole image rows)       */
    uint32_t src_x;  /* Image column of the first column of src                 */
} im2col_opt_fp16_spatz_params_t;

#endif /* IM2COL_OPT_FP16_SPATZ_PARAMS_H_ */
