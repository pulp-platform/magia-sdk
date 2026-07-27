#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "kernel_idma_utils.h"

#include "matmul_fp16_spatz.h"
#include "matmul_fp16_spatz_params.h"
#include "matmul_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "matmul_fp16_spatz"

static int alloc_l1(void **params, uint32_t M, uint32_t K, uint32_t O, uint32_t total_batches)
{
    volatile matmul_fp16_spatz_params_t *matmul_params;
    uintptr_t shard_A;
    uintptr_t shard_B;
    uintptr_t shard_Y;

    uint32_t batch_start;
    uint32_t batch_end;
    uint32_t batch_len;
    uint32_t shard;
    uint32_t left;

    shard = total_batches / NUM_HARTS;
    left  = total_batches % NUM_HARTS;

    batch_start = HID * shard + (HID < left ? HID : left);
    batch_end   = batch_start + shard + (HID < left ? 1 : 0);
    batch_len   = batch_end - batch_start;

    l1_alloc_init();

    matmul_params = l1_alloc(sizeof(matmul_fp16_spatz_params_t));
    if (!matmul_params)
        return ENOMEM;

    shard_A = l1_alloc(batch_len * (M * K) * sizeof(float16));
    if (!shard_A)
        return ENOMEM;

    shard_B = l1_alloc(batch_len * (K * O) * sizeof(float16));
    if (!shard_B)
        return ENOMEM;

    shard_Y = l1_alloc(batch_len * (M * O) * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    matmul_params->shard_A    = shard_A;
    matmul_params->shard_B    = shard_B;
    matmul_params->shard_Y    = shard_Y;
    matmul_params->M          = M;
    matmul_params->K          = K;
    matmul_params->O          = O;
    matmul_params->batch_start = batch_start;
    matmul_params->batch_len   = batch_len;

    *params = (void *) matmul_params;

    return 0;
}

static int init_input_params(void *params, const float16 *A, const float16 *B, uint32_t a_batched, uint32_t b_batched)
{
    volatile matmul_fp16_spatz_params_t *matmul_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t batch_start;
    uint32_t batch_len;
    uint32_t a_elems;
    uint32_t b_elems;

    matmul_params = (volatile matmul_fp16_spatz_params_t *) params;
    batch_start = matmul_params->batch_start;
    batch_len   = matmul_params->batch_len;
    a_elems     = matmul_params->M * matmul_params->K;
    b_elems     = matmul_params->K * matmul_params->O;

    if (batch_len == 0)
        return 0;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* batched == 1: this tile's batches are a contiguous L2 block -> a 1D transfer.
       batched == 0 (shared/broadcast weight): replicate the single 2D matrix into every
       batch slot with a 2D transfer whose source stride is 0. The Spatz task overwrites
       the whole output shard, so shard_Y is not zeroed here. */
    if (a_batched)
        idma_memcpy_1d(&idma_ctrl, 0, (uint32_t) (A + batch_start * a_elems), (uint32_t) matmul_params->shard_A, batch_len * a_elems * sizeof(float16));
    else
        idma_memcpy_2d(&idma_ctrl, 0, (uint32_t) A, (uint32_t) matmul_params->shard_A, a_elems * sizeof(float16), 0, batch_len);
    eu_idma_wait_a2o(&eu_ctrl, WFE);

    if (b_batched)
        idma_memcpy_1d(&idma_ctrl, 0, (uint32_t) (B + batch_start * b_elems), (uint32_t) matmul_params->shard_B, batch_len * b_elems * sizeof(float16));
    else
        idma_memcpy_2d(&idma_ctrl, 0, (uint32_t) B, (uint32_t) matmul_params->shard_B, b_elems * sizeof(float16), 0, batch_len);
    eu_idma_wait_a2o(&eu_ctrl, WFE);

    return 0;
}

static int offload_spatz_task(void *params)
{
    eu_controller_t eu_ctrl;
    int ret;

    eu_ctrl_init(&eu_ctrl);

    spatz_run_task_with_params(MATMUL_FP16_SPATZ_TASK, (uint32_t)params);

    ret = eu_spatz_wait(&eu_ctrl, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n", HID, KERNEL_NAME, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

static int store_result(void *params, float16 *Y)
{
    volatile matmul_fp16_spatz_params_t *matmul_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t batch_start;
    uint32_t batch_len;
    uint32_t y_elems;

    matmul_params = (volatile matmul_fp16_spatz_params_t *) params;
    batch_start = matmul_params->batch_start;
    batch_len = matmul_params->batch_len;
    y_elems = matmul_params->M * matmul_params->O;

    if (batch_len == 0)
        return 0;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* Y is always batched: this tile's batches are a contiguous L2 block -> a 1D transfer. */
    idma_memcpy_1d(&idma_ctrl, 1, (uint32_t) (Y + batch_start * y_elems), (uint32_t) matmul_params->shard_Y, batch_len * y_elems * sizeof(float16));
    eu_idma_wait_o2a(&eu_ctrl, WFE);

    return 0;
}

void MAGIA_matmul_fp16_spatz(const float16 *A, const float16 *B, float16 *Y, uint32_t M, uint32_t K, uint32_t O, uint32_t total_batches, uint32_t a_batched, uint32_t b_batched)
{
    int ret;
    volatile matmul_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, M, K, O, total_batches);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, A, B, a_batched, b_batched);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = offload_spatz_task((void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result((void *)params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
