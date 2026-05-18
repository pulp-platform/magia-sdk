#include <stdint.h>

#include "eventunit.h"
#include "tile.h"

#include "gemm_fp16_spatz.h"
#include "gemm_fp16_spatz_mem_layout.h"
#include "gemm_fp16_spatz_params.h"
#include "gemm_fp16_spatz_task_bin.h"

#define HID get_hartid()

static int init_input_params(void *params, const float16 *A, const float16 *B, const float16 *C, const float16 alpha, const float16 beta, int transA, int transB)
{
    volatile gemm_fp16_spatz_params_t *gemm_params;
    uint32_t shard;
    uint32_t left;
    uint32_t m_start;
    uint32_t m_end;
    uint32_t m_len;
    uint32_t local_idx;

    gemm_params = (volatile gemm_fp16_spatz_params_t *) params;

    shard = INPUT0_DIM0 / NUM_HARTS;
    left  = INPUT0_DIM0 % NUM_HARTS;

    m_start = HID * shard + (HID < left ? HID : left);
    m_end   = m_start + shard + (HID < left ? 1 : 0);
    m_len   = m_end - m_start;


    local_idx = 0;
    if (!transA) {
        for (int m = m_start; m < m_end; m++) {
            uint32_t row_base;
            row_base = m * INPUT0_DIM1;

            for (int k = 0; k < INPUT0_DIM1; k++) {
                uint32_t global_idx;
                uint32_t offset;

                global_idx = row_base + k;
                offset = local_idx * sizeof(float16);

                mmio_fp16(SHARD_A_BASE + offset) = A[global_idx];

                local_idx++;
            }
        }
    } else {
        for (int k = 0; k < INPUT0_DIM1; k++) {
            uint32_t row_base;
            row_base = k * INPUT0_DIM0;

            for (int m = m_start; m < m_end; m++) {
                uint32_t global_idx;
                uint32_t offset;

                global_idx = row_base + m;
                offset = local_idx * sizeof(float16);

                mmio_fp16(SHARD_A_BASE + offset) = A[global_idx];

                local_idx++;
            }
        }
    }

    if (!transB) {
        for (int i = 0; i < INPUT1_SIZE; i++) {
            uint32_t offset;
            offset = i * sizeof(float16);
            mmio_fp16(SHARD_B_BASE + offset) = B[i];
        }
    } else {
        for (int i = 0; i < INPUT1_SIZE; i++) {
            uint32_t offset;
            offset = i * sizeof(float16);
            mmio_fp16(SHARD_B_BASE + offset) = B[i];
        }
    }


    local_idx = 0;
    for (int m = m_start; m < m_end; m++) {
        uint32_t row_base;
        row_base = m * OUTPUT0_DIM1;
        for (int n = 0; n < OUTPUT0_DIM1; n++) {
            uint32_t global_idx;
            uint32_t offset;

            global_idx = row_base + n;
            offset = local_idx * sizeof(float16);

            mmio_fp16(SHARD_C_BASE + offset) = C[global_idx];
            mmio_fp16(SHARD_Y_BASE + offset) = 0;

            local_idx++;
        }
    }

    mmio_fp16(ALPHA_BASE) = alpha;
    mmio_fp16(BETA_BASE)  = beta;

    gemm_params->shard_A = SHARD_A_BASE;
    gemm_params->shard_B = SHARD_B_BASE;
    gemm_params->shard_C = SHARD_C_BASE;
    gemm_params->shard_Y = SHARD_Y_BASE;
    gemm_params->alpha = ALPHA_BASE;
    gemm_params->beta  = BETA_BASE;
    gemm_params->transA = transA;
    gemm_params->transB = transB;
    gemm_params->m_start = m_start;
    gemm_params->m_len   = m_len;
    gemm_params->M = INPUT0_DIM0;
    gemm_params->N = INPUT1_DIM0;
    gemm_params->K = INPUT0_DIM1;

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
    spatz_run_task_with_params(GEMM_FP16_SPATZ_TASK, GEMM_FP16_SPATZ_PARAMS_BASE);

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
    volatile gemm_fp16_spatz_params_t *gemm_params;
    uint32_t shard_Y_base;
    uint32_t m_start;
    uint32_t m_len;
    uint32_t local_idx;

    gemm_params = (volatile gemm_fp16_spatz_params_t *) params;

    shard_Y_base = gemm_params->shard_Y;

    m_start = gemm_params->m_start;
    m_len   = gemm_params->m_len;

    local_idx = 0;
    for (int m = 0; m < m_len; m++) {
        uint32_t global_m;
        uint32_t row_base;

        global_m = m_start + m;
        row_base = global_m * OUTPUT0_DIM1;
        for (int n = 0; n < OUTPUT0_DIM1; n++) {
            uint32_t global_idx;
            uint32_t offset;

            global_idx = row_base + n;
            offset = local_idx * sizeof(float16);
            Y[global_idx] = mmio_fp16(shard_Y_base + offset);

            local_idx++;
        }
    }

    return 0;
}

void MAGIA_gemm_fp16_spatz(const float16 *A, const float16 *B, const float16 *C, float16 alpha, float16 beta, int transA, int transB, float16 *Y)
{
    int ret;
    volatile gemm_fp16_spatz_params_t *params;

    params = (volatile gemm_fp16_spatz_params_t *) GEMM_FP16_SPATZ_PARAMS_BASE;

    ret = init_input_params(params, A, B, C, alpha, beta, transA, transB);
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
