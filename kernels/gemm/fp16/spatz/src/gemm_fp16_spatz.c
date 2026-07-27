#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "kernel_idma_utils.h"

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
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t m_start;
    uint32_t m_len;
    uint32_t K;
    uint32_t N;
    uint32_t b_bytes;

    gemm_params = (volatile gemm_fp16_spatz_params_t *) params;
    m_start = gemm_params->m_start;
    m_len   = gemm_params->m_len;
    K = gemm_params->K;
    N = gemm_params->N;
    b_bytes = B_shape[0] * B_shape[1] * sizeof(float16);

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    if (m_len > 0) {
        /* A: this tile's rows [m_start, m_end). transA == 0 -> A is [M, K], rows are
           contiguous (1D). transA == 1 -> A is [K, M], each of the K rows contributes
           m_len contiguous elements (starting at A[k*M + m_start]) spaced M apart: a
           strided 2D transfer packed contiguously into shard_A ([K, m_len]). */
        if (!gemm_params->transA) {
            idma_memcpy_1d(&idma_ctrl, 0, (uint32_t) (A + m_start * K), (uint32_t) gemm_params->shard_A, m_len * K * sizeof(float16));
            eu_idma_wait_a2o(&eu_ctrl, WFE);
        } else {
            idma_memcpy_2d(&idma_ctrl, 0, (uint32_t) (A + m_start), (uint32_t) gemm_params->shard_A,
                           m_len * sizeof(float16), A_shape[1] * sizeof(float16), A_shape[0]);
            eu_idma_wait_a2o(&eu_ctrl, WFE);
        }

        /* B: full matrix, contiguous. */
        idma_memcpy_1d(&idma_ctrl, 0, (uint32_t) B, (uint32_t) gemm_params->shard_B, b_bytes);
        eu_idma_wait_a2o(&eu_ctrl, WFE);

        /* C: this tile's rows [m_start, m_end) of [M, N] are contiguous. The Spatz task
           overwrites the whole output shard, so shard_Y is not zeroed here. */
        idma_memcpy_1d(&idma_ctrl, 0, (uint32_t) (C + m_start * N), (uint32_t) gemm_params->shard_C, m_len * N * sizeof(float16));
        eu_idma_wait_a2o(&eu_ctrl, WFE);
    }

    mmio_fp16(gemm_params->alpha) = alpha_val;
    mmio_fp16(gemm_params->beta)  = beta_val;

    return 0;
}

static int offload_spatz_task(void *params)
{
    eu_controller_t eu_ctrl;
    int ret;

    eu_ctrl_init(&eu_ctrl);

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
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t m_start;
    uint32_t m_len;
    uint32_t N;

    gemm_params = (volatile gemm_fp16_spatz_params_t *) params;
    m_start = gemm_params->m_start;
    m_len   = gemm_params->m_len;
    N = gemm_params->N;

    if (m_len == 0)
        return 0;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* This tile's output rows [m_start, m_end) of [M, N] are contiguous in L2. */
    idma_memcpy_1d(&idma_ctrl, 1, (uint32_t) (Y + m_start * N), (uint32_t) gemm_params->shard_Y, m_len * N * sizeof(float16));
    eu_idma_wait_o2a(&eu_ctrl, WFE);

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
