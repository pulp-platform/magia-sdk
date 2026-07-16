#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

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

static int init_input_params(void *params, const float16 *A, const float16 *B)
{
    volatile matmul_fp16_spatz_params_t *matmul_params;
    uintptr_t shard_A_base;
    uintptr_t shard_B_base;
    uintptr_t shard_Y_base;
    uint32_t local_idx;
    uint32_t batch_start;
    uint32_t batch_len;
    uint32_t M;
    uint32_t K;
    uint32_t O;

    matmul_params = (volatile matmul_fp16_spatz_params_t *) params;
    shard_A_base = matmul_params->shard_A;
    shard_B_base = matmul_params->shard_B;
    shard_Y_base = matmul_params->shard_Y;

    batch_start = matmul_params->batch_start;
    batch_len   = matmul_params->batch_len;
    M           = matmul_params->M;
    K           = matmul_params->K;
    O           = matmul_params->O;

    local_idx = 0;
    for (uint32_t b = 0; b < batch_len; b++) {
        uint32_t offset_A_2d = (batch_start + b) * (M * K);
        for (uint32_t m = 0; m < M; m++) {
            uint32_t row_base = m * K;
            for (uint32_t k = 0; k < K; k++) {
                uint32_t global_idx = offset_A_2d + row_base + k;
                uint32_t offset = local_idx * sizeof(float16);
                mmio_fp16(shard_A_base + offset) = A[global_idx];
                local_idx++;
            }
        }
    }

    local_idx = 0;
    for (uint32_t b = 0; b < batch_len; b++) {
        uint32_t offset_B_2d = (batch_start + b) * (K * O);
        for (uint32_t k = 0; k < K; k++) {
            uint32_t row_base = k * O;
            for (uint32_t o = 0; o < O; o++) {
                uint32_t global_idx = offset_B_2d + row_base + o;
                uint32_t offset = local_idx * sizeof(float16);
                mmio_fp16(shard_B_base + offset) = B[global_idx];
                local_idx++;
            }
        }
    }

    local_idx = 0;
    for (uint32_t b = 0; b < batch_len; b++) {
        for (uint32_t m = 0; m < M; m++) {
            for (uint32_t o = 0; o < O; o++) {
                uint32_t offset = local_idx * sizeof(float16);
                mmio_fp16(shard_Y_base + offset) = 0;
                local_idx++;
            }
        }
    }

    return 0;
}

static int offload_spatz_task(void *params)
{
    eu_controller_t eu_ctrl;
    eu_config_t eu_cfg;
    int ret;

    eu_cfg.hartid = HID;
    eu_ctrl.base = NULL;
    eu_ctrl.cfg = &eu_cfg;
    eu_ctrl.api = &eu_api;

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
    uintptr_t shard_Y_base;
    uint32_t batch_start;
    uint32_t batch_len;
    uint32_t local_idx;
    uint32_t M;
    uint32_t O;

    matmul_params = (volatile matmul_fp16_spatz_params_t *) params;
    shard_Y_base = matmul_params->shard_Y;
    batch_start = matmul_params->batch_start;
    batch_len = matmul_params->batch_len;
    M = matmul_params->M;
    O = matmul_params->O;

    local_idx = 0;
    for (uint32_t b = 0; b < batch_len; b++) {
        uint32_t offset_Y_2d = (batch_start + b) * (M * O);
        for (uint32_t m = 0; m < M; m++) {
            uint32_t row_base = m * O;
            for (uint32_t o = 0; o < O; o++) {
                uint32_t global_idx = offset_Y_2d + row_base + o;
                uint32_t offset = local_idx * sizeof(float16);
                Y[global_idx] = mmio_fp16(shard_Y_base + offset);
                local_idx++;
            }
        }
    }

    return 0;
}

void MAGIA_matmul_fp16_spatz(const float16 *A, const float16 *B, float16 *Y, uint32_t M, uint32_t K, uint32_t O, uint32_t total_batches)
{
    int ret;
    volatile matmul_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, M, K, O, total_batches);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, A, B);
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
