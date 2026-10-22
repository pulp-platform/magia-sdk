#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "rmsnorm_fp16_spatz.h"
#include "rmsnorm_fp16_spatz_params.h"
#include "rmsnorm_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "rmsnorm_fp16_spatz"

static int alloc_l1(void **params, const float16 epsilon, uint32_t rows, uint32_t w_len)
{
    volatile rmsnorm_fp16_spatz_params_t *rms_params;
    uint32_t shard_X;
    uint32_t gamma;
    uint32_t eps;

    size_t r_start;
    size_t r_end;
    size_t r_len;
    size_t shard;
    size_t left;

    shard = rows / NUM_HARTS;
    left  = rows % NUM_HARTS;

    r_start = HID * shard + (HID < left ? HID : left);
    r_end   = r_start + shard + (HID < left ? 1 : 0);
    r_len   = r_end - r_start;

    l1_alloc_init();

    rms_params = l1_alloc(sizeof(rmsnorm_fp16_spatz_params_t));
    if (!rms_params)
        return ENOMEM;

    /* The task normalizes the rows in place: one shard for X and Y. */
    shard_X = (uint32_t)l1_alloc(r_len * w_len * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    gamma = (uint32_t)l1_alloc(w_len * sizeof(float16));
    if (!gamma)
        return ENOMEM;

    eps = (uint32_t)l1_alloc(sizeof(float16));
    if (!eps)
        return ENOMEM;

    /* epsilon is a single element: the CV32 copies it to L1 itself, no iDMA needed. */
    mmio_fp16(eps) = epsilon;

    rms_params->shard_X = shard_X;
    rms_params->shard_Y = shard_X;
    rms_params->gamma   = gamma;
    rms_params->eps     = eps;
    rms_params->r_start = r_start;
    rms_params->r_len   = r_len;
    rms_params->w_len   = w_len;

    *params = (void *)rms_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X, const float16 *scale)
{
    volatile rmsnorm_fp16_spatz_params_t *rms_params;
    uint32_t r_start;
    uint32_t r_len;
    uint32_t w_len;

    rms_params = (volatile rmsnorm_fp16_spatz_params_t *)params;
    r_start    = rms_params->r_start;
    r_len      = rms_params->r_len;
    w_len      = rms_params->w_len;

    if (r_len == 0)
        return 0;

    /* This tile's rows [r_start, r_start+r_len) are contiguous in X. */
    idma_memcpy_1d(0,
                   (uint32_t)(X + r_start * w_len),
                   (uint32_t)rms_params->shard_X,
                   r_len * w_len * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    idma_memcpy_1d(0, (uint32_t)scale, (uint32_t)rms_params->gamma, w_len * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    return 0;
}

static int offload_spatz_task(void *params)
{
    int ret;

    spatz_run_task_with_params(RMSNORM_FP16_SPATZ_TASK, (uint32_t)params);

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

/* RMSNorm of rows already in this tile's L1: no sharding across tiles, no L1 allocation and no
   L2 transfers. Y may alias X. */
void MAGIA_rmsnorm_fp16_spatz_l1(const float16 *X,
                                 const float16 *scale,
                                 const float16 *epsilon,
                                 uint32_t rows,
                                 uint32_t w_len,
                                 float16 *Y)
{
    volatile rmsnorm_fp16_spatz_params_t params = {
        .shard_X = (uintptr_t)X,
        .shard_Y = (uintptr_t)Y,
        .gamma   = (uintptr_t)scale,
        .eps     = (uintptr_t)epsilon,
        .r_start = 0,
        .r_len   = rows,
        .w_len   = w_len,
    };
    int ret;

    ret = offload_spatz_task((void *)&params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
    }
}

static int store_result(void *params, float16 *Y)
{
    volatile rmsnorm_fp16_spatz_params_t *rms_params;
    uint32_t r_start;
    uint32_t r_len;
    uint32_t w_len;

    rms_params = (volatile rmsnorm_fp16_spatz_params_t *)params;
    r_start    = rms_params->r_start;
    r_len      = rms_params->r_len;
    w_len      = rms_params->w_len;

    if (r_len == 0)
        return 0;

    /* This tile's output rows are contiguous in Y. */
    idma_memcpy_1d(1,
                   (uint32_t)(Y + r_start * w_len),
                   (uint32_t)rms_params->shard_Y,
                   r_len * w_len * sizeof(float16));
    eu_idma_wait_o2a(WFE);

    return 0;
}

void MAGIA_rmsnorm_fp16_spatz(const float16 *X,
                              const float16 *scale,
                              const float16 epsilon,
                              uint32_t rows,
                              uint32_t w_len,
                              float16 *Y)
{
    int ret;
    volatile rmsnorm_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, epsilon, rows, w_len);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, X, scale);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    ret = offload_spatz_task((void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    ret = store_result((void *)params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
