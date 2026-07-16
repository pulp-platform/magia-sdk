#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

#include "gemm_fp16_spatz.h"
#include "gemm_fp16_spatz_params.h"
#include "gemm_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "gemm_fp16_spatz"

static int alloc_l1(void **params, uint32_t A_shape[2], uint32_t B_shape[2], uint32_t C_shape[2], uint32_t Y_shape[2], int transA, int transB)
{
    volatile gemm_fp16_spatz_params_t *gemm_params;
    uintptr_t shard_A;
    uintptr_t shard_B;
    uintptr_t shard_C;
    uintptr_t shard_Y;
    uintptr_t alpha;
    uintptr_t beta;

    size_t m;
    size_t n;
    size_t k;
    size_t shard;
    size_t left;
    size_t m_start;
    size_t m_end;
    size_t m_len;
    size_t shard_A_len;
    size_t shard_B_len;
    size_t shard_C_len;
    size_t shard_Y_len;

    m = Y_shape[0];
    n = Y_shape[1];
    k = (!transA) ? A_shape[1] : A_shape[0];

    shard = m / NUM_HARTS;
    left = m % NUM_HARTS;

    m_start = HID * shard + (HID < left ? HID : left);
    m_end = m_start + shard + (HID < left ? 1 : 0);
    m_len = m_end - m_start;

    shard_A_len = m_len * k;
    shard_B_len = k * n;
    shard_C_len = m_len * n;
    shard_Y_len = m_len * n;

    l1_alloc_init();

    gemm_params = l1_alloc(sizeof(gemm_fp16_spatz_params_t));
    if (!gemm_params)
        return ENOMEM;

    shard_A = l1_alloc(shard_A_len * sizeof(float16));
    if (!shard_A)
        return ENOMEM;

    shard_B = l1_alloc(shard_B_len * sizeof(float16));
    if (!shard_B)
        return ENOMEM;

    shard_C = l1_alloc(shard_C_len * sizeof(float16));
    if (!shard_C)
        return ENOMEM;

    shard_Y = l1_alloc(shard_Y_len * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    alpha = l1_alloc(sizeof(float16));
    if (!alpha)
        return ENOMEM;

    beta = l1_alloc(sizeof(float16));
    if (!beta)
        return ENOMEM;

    gemm_params->shard_A = shard_A;
    gemm_params->shard_B = shard_B;
    gemm_params->shard_C = shard_C;
    gemm_params->shard_Y = shard_Y;
    gemm_params->alpha   = alpha;
    gemm_params->beta    = beta;
    gemm_params->transA  = transA;
    gemm_params->transB  = transB;
    gemm_params->m_start = (uint32_t) m_start;
    gemm_params->m_len   = (uint32_t) m_len;
    gemm_params->M       = (uint32_t) m;
    gemm_params->N       = (uint32_t) n;
    gemm_params->K       = (uint32_t) k;

    *params = (void *) gemm_params;

    return 0;
}

static int init_input_params(void *params, const float16 *A, const float16 *B, const float16 *C, const float16 alpha_val, const float16 beta_val, uint32_t A_shape[2], uint32_t B_shape[2], uint32_t C_shape[2], uint32_t Y_shape[2])
{
    volatile gemm_fp16_spatz_params_t *gemm_params;
    uintptr_t shard_A;
    uintptr_t shard_B;
    uintptr_t shard_C;
    uintptr_t shard_Y;
    uint32_t m_start;
    uint32_t m_end;
    uint32_t m_len;
    uint32_t local_idx;
    uint32_t b_size;

    gemm_params = (volatile gemm_fp16_spatz_params_t *) params;
    shard_A = gemm_params->shard_A;
    shard_B = gemm_params->shard_B;
    shard_C = gemm_params->shard_C;
    shard_Y = gemm_params->shard_Y;
    m_start = gemm_params->m_start;
    m_len   = gemm_params->m_len;
    m_end   = m_start + m_len;

    local_idx = 0;
    if (!gemm_params->transA) {
        for (uint32_t m = m_start; m < m_end; m++) {
            uint32_t row_base = m * gemm_params->K;
            for (uint32_t k = 0; k < gemm_params->K; k++) {
                uint32_t global_idx = row_base + k;
                uint32_t offset = local_idx * sizeof(float16);
                mmio_fp16(shard_A + offset) = A[global_idx];
                local_idx++;
            }
        }
    } else {
        for (uint32_t k = 0; k < A_shape[0]; k++) {
            for (uint32_t m = 0; m < m_len; m++) {
                uint32_t m_global = m_start + m;
                uint32_t global_idx = k * A_shape[1] + m_global;
                uint32_t offset = local_idx * sizeof(float16);
                mmio_fp16(shard_A + offset) = A[global_idx];
                local_idx++;
            }
        }
    }

    b_size = B_shape[0] * B_shape[1];
    for (uint32_t i = 0; i < b_size; i++) {
        uint32_t offset = i * sizeof(float16);
        mmio_fp16(shard_B + offset) = B[i];
    }

    local_idx = 0;
    for (uint32_t m = m_start; m < m_end; m++) {
        uint32_t row_base = m * gemm_params->N;
        for (uint32_t n = 0; n < gemm_params->N; n++) {
            uint32_t global_idx = row_base + n;
            uint32_t offset = local_idx * sizeof(float16);
            mmio_fp16(shard_C + offset) = C[global_idx];
            mmio_fp16(shard_Y + offset) = 0;
            local_idx++;
        }
    }

    mmio_fp16(gemm_params->alpha) = alpha_val;
    mmio_fp16(gemm_params->beta)  = beta_val;

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

    spatz_run_task_with_params(GEMM_FP16_SPATZ_TASK, (uint32_t)params);

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
    volatile gemm_fp16_spatz_params_t *gemm_params;
    uintptr_t shard_Y_base;
    uint32_t m_start;
    uint32_t m_len;
    uint32_t local_idx;

    gemm_params = (volatile gemm_fp16_spatz_params_t *) params;
    shard_Y_base = gemm_params->shard_Y;
    m_start = gemm_params->m_start;
    m_len   = gemm_params->m_len;

    local_idx = 0;
    for (uint32_t m = 0; m < m_len; m++) {
        uint32_t global_m = m_start + m;
        uint32_t row_base = global_m * gemm_params->N;
        for (uint32_t n = 0; n < gemm_params->N; n++) {
            uint32_t global_idx = row_base + n;
            uint32_t offset = local_idx * sizeof(float16);
            Y[global_idx] = mmio_fp16(shard_Y_base + offset);
            local_idx++;
        }
    }

    return 0;
}

void MAGIA_gemm_fp16_spatz(const float16 *A, const float16 *B, const float16 *C, float16 alpha, float16 beta, int transA, int transB, uint32_t A_shape[2], uint32_t B_shape[2], uint32_t C_shape[2], uint32_t Y_shape[2], float16 *Y)
{
    int ret;
    volatile gemm_fp16_spatz_params_t *params;

    ret = alloc_l1(&params, A_shape, B_shape, C_shape, Y_shape, transA, transB);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params(params, A, B, C, alpha, beta, A_shape, B_shape, C_shape, Y_shape);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = offload_spatz_task(params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result(params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
