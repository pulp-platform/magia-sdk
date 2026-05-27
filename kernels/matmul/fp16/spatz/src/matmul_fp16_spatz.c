#include <stdint.h>

#include "eventunit.h"
#include "tile.h"

#include "matmul_fp16_spatz.h"
#include "matmul_fp16_spatz_mem_layout.h"
#include "matmul_fp16_spatz_params.h"
#include "matmul_fp16_spatz_task_bin.h"

#define HID get_hartid()

static int init_input_params(void *params, const float16 *A, const float16 *B, uint32_t M, uint32_t K, uint32_t O, uint32_t total_batches)
{
    volatile matmul_fp16_spatz_params_t *matmul_params;
    uint32_t batch_start;
    uint32_t batch_end;
    uint32_t batch_len;
    uint32_t local_idx;
    uint32_t shard;
    uint32_t left;

    matmul_params = (volatile matmul_fp16_spatz_params_t *) params;

    shard = total_batches / NUM_HARTS;
    left  = total_batches % NUM_HARTS;

    batch_start = HID * shard + (HID < left ? HID : left);
    batch_end   = batch_start + shard + (HID < left ? 1 : 0);
    batch_len   = batch_end - batch_start;


    /* Matrix A */
    local_idx = 0;
    for (uint32_t b = 0; b < batch_len; b++) {
        uint32_t offset_A_2d = (batch_start + b) * (M * K);
        for (uint32_t m = 0; m < M; m++) {
            uint32_t row_base = m * K;
            for (uint32_t k = 0; k < K; k++) {
                uint32_t global_idx = offset_A_2d + row_base + k;
                uint32_t offset = local_idx * sizeof(float16);
                mmio_fp16(SHARD_A_BASE + offset) = A[global_idx];
                local_idx++;
            }
        }
    }
    /* Matrix B */
    local_idx = 0;
    for (uint32_t b = 0; b < batch_len; b++) {
        uint32_t offset_B_2d = (batch_start + b) * (K * O);
        for (uint32_t k = 0; k < K; k++) {
            uint32_t row_base = k * O;
            for (uint32_t o = 0; o < O; o++) {
                uint32_t global_idx = offset_B_2d + row_base + o;
                uint32_t offset = local_idx * sizeof(float16);
                mmio_fp16(SHARD_B_BASE + offset) = B[global_idx];
                local_idx++;
            }
        }
    }
    /* Result Matrix */
    local_idx = 0;
    for (uint32_t b = 0; b < batch_len; b++) {
        for (uint32_t m = 0; m < M; m++) {
            for (uint32_t o = 0; o < O; o++) {
                uint32_t offset = local_idx * sizeof(float16);
                mmio_fp16(SHARD_Y_BASE + offset) = 0;
                local_idx++;
            }
        }
    }


    matmul_params->shard_A = SHARD_A_BASE;
    matmul_params->shard_B = SHARD_B_BASE;
    matmul_params->shard_Y = SHARD_Y_BASE;
    matmul_params->M = M;
    matmul_params->K = K;
    matmul_params->O = O;
    matmul_params->batch_start = batch_start;
    matmul_params->batch_len = batch_len;

    return 0;
}

static int offload_spatz_task()
{
    eu_controller_t eu_ctrl;
    eu_config_t eu_cfg;
    int ret;

    eu_cfg.hartid = HID;
    eu_ctrl.base = NULL;
    eu_ctrl.cfg = &eu_cfg;
    eu_ctrl.api = &eu_api;

    spatz_init(SPATZ_BINARY_START);
    spatz_run_task_with_params(MATMUL_FP16_SPATZ_TASK, MATMUL_FP16_SPATZ_PARAMS_BASE);

    ret = eu_spatz_wait(&eu_ctrl, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] Wait on Spatz task completion failed with error: %d\n", HID, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();
    spatz_clk_dis();

exit:
    return ret;
}

static int store_result(void *params, float16 *Y)
{
    volatile matmul_fp16_spatz_params_t *matmul_params;
    uint32_t shard_Y_base;
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

    params = (volatile matmul_fp16_spatz_params_t *) MATMUL_FP16_SPATZ_PARAMS_BASE;

    ret = init_input_params(params, A, B, M, K, O, total_batches);
    if (ret != 0) {
        printf("[CV32 (%d)] Params initialization failed with error: %d\n", HID, ret);
        return;
    }

    ret = offload_spatz_task();
    if (ret != 0) {
        printf("[CV32 (%d)] Spatz task offloading failed with error: %d\n", HID, ret);
        return;
    }

    ret = store_result(params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] Result write back failed with error: %d\n", HID, ret);
    }
}
