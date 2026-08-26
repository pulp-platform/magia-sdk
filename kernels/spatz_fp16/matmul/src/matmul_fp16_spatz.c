#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "kernels_dma_utils.h"

#include "matmul_fp16_spatz.h"
#include "matmul_fp16_spatz_params.h"
#include "matmul_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "matmul_fp16_spatz"

/*
 * The rows of A are sharded, not the batches, and every tile runs every batch. Sharding
 * the batches instead leaves all the work on tile 0 whenever the batch count is 1 - which
 * is the common case for a matmul against a weight - and leaves most of a 64-tile mesh
 * idle for the 5- and 10-way batched attention shapes.
 */
static int alloc_l1(void **params, uint32_t M, uint32_t K, uint32_t O, uint32_t total_batches, uint32_t a_batched, uint32_t b_batched)
{
    volatile matmul_fp16_spatz_params_t *matmul_params;
    uintptr_t shard_A;
    uintptr_t shard_B;
    uintptr_t shard_Y;

    uint32_t a_batches;
    uint32_t b_batches;
    uint32_t m_start;
    uint32_t m_end;
    uint32_t m_len;
    uint32_t shard;
    uint32_t left;

    shard = M / NUM_HARTS;
    left  = M % NUM_HARTS;

    m_start = HID * shard + (HID < left ? HID : left);
    m_end   = m_start + shard + (HID < left ? 1 : 0);
    m_len   = m_end - m_start;

    /* A shared operand is staged once, not once per batch. */
    a_batches = a_batched ? total_batches : 1;
    b_batches = b_batched ? total_batches : 1;

    l1_alloc_init();

    matmul_params = l1_alloc(sizeof(matmul_fp16_spatz_params_t));
    if (!matmul_params)
        return ENOMEM;

    shard_A = l1_alloc(a_batches * (m_len * K) * sizeof(float16));
    if (!shard_A)
        return ENOMEM;

    shard_B = l1_alloc(b_batches * (K * O) * sizeof(float16));
    if (!shard_B)
        return ENOMEM;

    shard_Y = l1_alloc(total_batches * (m_len * O) * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    matmul_params->shard_A   = shard_A;
    matmul_params->shard_B   = shard_B;
    matmul_params->shard_Y   = shard_Y;
    matmul_params->M         = m_len;
    matmul_params->K         = K;
    matmul_params->O         = O;
    matmul_params->M_total   = M;
    matmul_params->m_start   = m_start;
    matmul_params->batch_len = total_batches;
    matmul_params->a_batched = a_batched;
    matmul_params->b_batched = b_batched;
    matmul_params->shard_bias = 0u;
    matmul_params->bias_mode = MATMUL_BIAS_NONE;
    matmul_params->bias_batched = 0u;
    matmul_params->transpose_b = 0u;

    *params = (void *) matmul_params;

    return 0;
}

/*
 * A: this tile's rows are a rectangle - one run of m_len * K elements per batch, M_total *
 * K apart in L2 and m_len * K apart in L1. B is not sharded and is contiguous, so one
 * transfer covers every batch of it. There is no shard_Y prefill: the task writes every
 * element of the rows this tile owns.
 */
static int init_input_params(void *params, kdma_t *d, const float16 *A, const float16 *B)
{
    volatile matmul_fp16_spatz_params_t *matmul_params;
    uint32_t a_batches;
    uint32_t b_batches;
    uint32_t m_len;
    uint32_t K;
    uint32_t O;

    matmul_params = (volatile matmul_fp16_spatz_params_t *) params;
    m_len = matmul_params->M;
    K     = matmul_params->K;
    O     = matmul_params->O;

    a_batches = matmul_params->a_batched ? matmul_params->batch_len : 1;
    b_batches = matmul_params->b_batched ? matmul_params->batch_len : 1;

    kdma_in_2d(d,
               (uintptr_t)(A + (uintptr_t)matmul_params->m_start * K),
               matmul_params->shard_A,
               m_len * K * sizeof(float16),
               matmul_params->M_total * K * sizeof(float16),
               m_len * K * sizeof(float16),
               a_batches);

    kdma_in(d, (uintptr_t)B, matmul_params->shard_B, b_batches * K * O * sizeof(float16));

    return 0;
}

static int offload_spatz_task(kdma_t *d, void *params)
{
    int ret;

    spatz_run_task_with_params(MATMUL_FP16_SPATZ_TASK, (uint32_t)params);

    ret = eu_spatz_wait(&d->eu, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n", HID, KERNEL_NAME, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

/* Mirror of A's staging: the same rectangle, back out. */
static int store_result(void *params, kdma_t *d, float16 *Y)
{
    volatile matmul_fp16_spatz_params_t *matmul_params;
    uint32_t m_len;
    uint32_t O;

    matmul_params = (volatile matmul_fp16_spatz_params_t *) params;
    m_len = matmul_params->M;
    O     = matmul_params->O;

    kdma_out_2d(d,
                (uintptr_t)(Y + (uintptr_t)matmul_params->m_start * O),
                matmul_params->shard_Y,
                m_len * O * sizeof(float16),
                matmul_params->M_total * O * sizeof(float16),
                m_len * O * sizeof(float16),
                matmul_params->batch_len);

    return 0;
}

void MAGIA_matmul_fp16_spatz(const float16 *A, const float16 *B, float16 *Y, uint32_t M, uint32_t K, uint32_t O, uint32_t total_batches, uint32_t a_batched, uint32_t b_batched)
{
    int ret;
    volatile matmul_fp16_spatz_params_t *params;
    kdma_t d;

    ret = alloc_l1((void **)&params, M, K, O, total_batches, a_batched, b_batched);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    /* More tiles than rows: nothing staged, nothing offloaded, nothing written back. */
    if (params->M == 0 || params->batch_len == 0)
        return;

    kdma_open(&d);

    ret = init_input_params((void *)params, &d, A, B);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = offload_spatz_task(&d, (void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result((void *)params, &d, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
