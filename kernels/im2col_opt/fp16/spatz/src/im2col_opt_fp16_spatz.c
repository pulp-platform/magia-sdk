#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "im2col_opt_fp16_spatz.h"
#include "im2col_opt_fp16_spatz_params.h"
#include "im2col_opt_fp16_spatz_task_bin.h"

#define HID                        get_hartid()
#define KERNEL_NAME                "im2col_opt_fp16_spatz"

/* Output size along a side of n pixels. */
#define OUT_SIZE(n, ksize, stride) (((n) + 2 * (((ksize) - 1) / 2) - (ksize)) / (stride) + 1)

/* This tile's output rows [*row_start, *row_start + *row_len) out of rows. */
static void tile_rows(uint32_t rows, uint32_t *row_start, uint32_t *row_len)
{
    uint32_t elems = rows / NUM_HARTS;
    uint32_t left  = rows % NUM_HARTS;

    *row_start = HID * elems + (HID < left ? HID : left);
    *row_len   = elems + (HID < left ? 1 : 0);
}

/* Image rows [*lo, *hi) read by the output rows [row_start, row_start + row_len). */
static void input_rows(uint32_t height,
                       uint32_t ksize,
                       uint32_t stride,
                       uint32_t row_start,
                       uint32_t row_len,
                       uint32_t *lo,
                       uint32_t *hi)
{
    int32_t pad   = (ksize - 1) / 2;
    int32_t first = (int32_t)(row_start * stride) - pad;
    int32_t last  = (int32_t)((row_start + row_len - 1) * stride) - pad + (int32_t)ksize;

    *lo = first < 0 ? 0 : (uint32_t)first;
    *hi = last > (int32_t)height ? height : (last < 0 ? 0 : (uint32_t)last);
    if (row_len == 0 || *hi < *lo)
        *hi = *lo;
}

static int alloc_l1(
    void **params, uint32_t height, uint32_t width, uint32_t cin, uint32_t ksize, uint32_t stride)
{
    volatile im2col_opt_fp16_spatz_params_t *ic_params;
    uintptr_t src;
    uintptr_t dst;
    uint32_t out_h;
    uint32_t out_w;
    uint32_t row_start;
    uint32_t row_len;
    uint32_t lo;
    uint32_t hi;

    out_h = OUT_SIZE(height, ksize, stride);
    out_w = OUT_SIZE(width, ksize, stride);
    tile_rows(out_h, &row_start, &row_len);
    input_rows(height, ksize, stride, row_start, row_len, &lo, &hi);

    l1_alloc_init();

    ic_params = l1_alloc(sizeof(im2col_opt_fp16_spatz_params_t));
    if (!ic_params)
        return ENOMEM;

    src = (uintptr_t)l1_alloc((hi - lo) * width * cin * sizeof(float16));
    if (!src)
        return ENOMEM;

    dst = (uintptr_t)l1_alloc(row_len * out_w * ksize * ksize * cin * sizeof(float16));
    if (!dst)
        return ENOMEM;

    ic_params->src    = src;
    ic_params->dst    = dst;
    ic_params->height = height;
    ic_params->width  = width;
    ic_params->cin    = cin;
    ic_params->ksize  = ksize;
    ic_params->stride = stride;
    ic_params->pad    = (ksize - 1) / 2;
    ic_params->row    = row_start;
    ic_params->lo     = lo;
    ic_params->x0     = 0;
    ic_params->cp     = out_w;
    ic_params->src_w  = width;
    ic_params->src_x  = 0;

    *params = (void *)ic_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X)
{
    volatile im2col_opt_fp16_spatz_params_t *ic_params;
    uint32_t row_start;
    uint32_t row_len;
    uint32_t lo;
    uint32_t hi;
    uint32_t pitch;

    ic_params = (volatile im2col_opt_fp16_spatz_params_t *)params;
    pitch     = ic_params->width * ic_params->cin;
    tile_rows(
        OUT_SIZE(ic_params->height, ic_params->ksize, ic_params->stride), &row_start, &row_len);
    input_rows(
        ic_params->height, ic_params->ksize, ic_params->stride, row_start, row_len, &lo, &hi);

    if (hi == lo)
        return 0;

    /* The image rows read by this tile's output rows, whole and contiguous in X. */
    idma_memcpy_1d(0,
                   (uint32_t)(X + lo * pitch),
                   (uint32_t)ic_params->src,
                   (hi - lo) * pitch * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    return 0;
}

static int offload_spatz_task(void *params)
{
    int ret;

    spatz_run_task_with_params(IM2COL_OPT_FP16_SPATZ_TASK, (uint32_t)params);

    ret = eu_spatz_wait(WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

void MAGIA_im2col_opt_fp16_spatz_l1(const im2col_opt_fp16_spatz_params_t *params)
{
    int ret;

    ret = offload_spatz_task((void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
    }
}

static int store_result(void *params, float16 *Y)
{
    volatile im2col_opt_fp16_spatz_params_t *ic_params;
    uint32_t row_start;
    uint32_t row_len;
    uint32_t row_size;

    ic_params = (volatile im2col_opt_fp16_spatz_params_t *)params;
    row_size  = ic_params->cp * ic_params->ksize * ic_params->ksize * ic_params->cin;
    tile_rows(
        OUT_SIZE(ic_params->height, ic_params->ksize, ic_params->stride), &row_start, &row_len);

    if (row_len == 0)
        return 0;

    /* The patches of this tile's output rows are contiguous in Y. */
    idma_memcpy_1d(1,
                   (uint32_t)(Y + row_start * row_size),
                   (uint32_t)ic_params->dst,
                   row_len * row_size * sizeof(float16));
    eu_idma_wait_o2a(WFE);

    return 0;
}

void MAGIA_im2col_opt_fp16_spatz(const float16 *X,
                                 float16 *Y,
                                 uint32_t height,
                                 uint32_t width,
                                 uint32_t cin,
                                 uint32_t ksize,
                                 uint32_t stride)
{
    int ret;
    volatile im2col_opt_fp16_spatz_params_t *params;
    uintptr_t dst;
    uint32_t row_start;
    uint32_t row_len;
    uint32_t row_size;

    ret = alloc_l1((void **)&params, height, width, cin, ksize, stride);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, X);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    /* The task builds one output row: one task per row of this tile. */
    tile_rows(OUT_SIZE(height, ksize, stride), &row_start, &row_len);
    row_size = params->cp * ksize * ksize * cin;
    dst      = params->dst;
    for (uint32_t r = 0; r < row_len; r++) {
        params->row = row_start + r;
        params->dst = dst + r * row_size * sizeof(float16);

        ret = offload_spatz_task((void *)params);
        if (ret != 0) {
            printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n",
                   HID,
                   KERNEL_NAME,
                   ret);
            return;
        }
    }
    params->dst = dst;

    ret = store_result((void *)params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
