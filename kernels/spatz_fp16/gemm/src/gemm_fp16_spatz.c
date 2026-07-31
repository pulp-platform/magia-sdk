#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

#include "kernels_dma_utils.h"

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

static int init_input_params(kdma_t *d, void *params, const float16 *A, const float16 *B, const float16 *C, const float16 alpha_val, const float16 beta_val, uint32_t A_shape[2], uint32_t B_shape[2], uint32_t C_shape[2], uint32_t Y_shape[2])
{
    volatile gemm_fp16_spatz_params_t *gemm_params;
    uintptr_t shard_A;
    uintptr_t shard_B;
    uintptr_t shard_C;
    uint32_t m_start;
    uint32_t m_len;
    uint32_t b_size;

    gemm_params = (volatile gemm_fp16_spatz_params_t *) params;
    shard_A = gemm_params->shard_A;
    shard_B = gemm_params->shard_B;
    shard_C = gemm_params->shard_C;
    m_start = gemm_params->m_start;
    m_len   = gemm_params->m_len;

    /* A: without transA this tile's rows m_start..m_end are one contiguous run. With
     * transA it owns a column slice of an [K, M] matrix, which is a rectangle: K rows of
     * m_len elements, m_len apart in L1 and A_shape[1] apart in L2. */
    if (!gemm_params->transA)
        kdma_in(d,
                (uintptr_t) (A + (uintptr_t) m_start * gemm_params->K),
                shard_A,
                m_len * gemm_params->K * sizeof(float16));
    else
        kdma_in_2d(d,
                   (uintptr_t) (A + m_start),
                   shard_A,
                   m_len * sizeof(float16),
                   A_shape[1] * sizeof(float16),
                   m_len * sizeof(float16),
                   A_shape[0]);

    /* B is not sharded - every tile needs all of it. */
    b_size = B_shape[0] * B_shape[1];
    kdma_in(d, (uintptr_t) B, shard_B, b_size * sizeof(float16));

    /* C: the same rows as A, so again one contiguous run. shard_Y is not pre-cleared,
     * the task writes every element of the rows this tile owns. */
    kdma_in(d,
            (uintptr_t) (C + (uintptr_t) m_start * gemm_params->N),
            shard_C,
            m_len * gemm_params->N * sizeof(float16));

    mmio_fp16(gemm_params->alpha) = alpha_val;
    mmio_fp16(gemm_params->beta)  = beta_val;

    return 0;
}

static int offload_spatz_task(kdma_t *d, void *params)
{
    int ret;

    spatz_run_task_with_params(GEMM_FP16_SPATZ_TASK, (uint32_t)params);

    ret = eu_spatz_wait(&d->eu, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n", HID, KERNEL_NAME, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

static int store_result(kdma_t *d, void *params, float16 *Y)
{
    volatile gemm_fp16_spatz_params_t *gemm_params;

    gemm_params = (volatile gemm_fp16_spatz_params_t *) params;

    kdma_out(d,
             (uintptr_t) (Y + (uintptr_t) gemm_params->m_start * gemm_params->N),
             gemm_params->shard_Y,
             gemm_params->m_len * gemm_params->N * sizeof(float16));

    return 0;
}

void MAGIA_gemm_fp16_spatz(const float16 *A, const float16 *B, const float16 *C, float16 alpha, float16 beta, int transA, int transB, uint32_t A_shape[2], uint32_t B_shape[2], uint32_t C_shape[2], uint32_t Y_shape[2], float16 *Y)
{
    int ret;
    volatile gemm_fp16_spatz_params_t *params;
    kdma_t d;

    kdma_open(&d);

    ret = alloc_l1(&params, A_shape, B_shape, C_shape, Y_shape, transA, transB);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    /* More tiles than rows of Y: nothing staged, nothing offloaded, nothing written back.
     * The task already returns early on an empty shard, but the zero-length transfers
     * around it have no reason to be issued. */
    if (params->m_len == 0)
        return;

    ret = init_input_params(&d, (void *)params, A, B, C, alpha, beta, A_shape, B_shape, C_shape, Y_shape);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = offload_spatz_task(&d, (void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result(&d, (void *)params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
