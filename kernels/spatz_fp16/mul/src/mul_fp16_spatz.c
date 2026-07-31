#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "kernels_dma_utils.h"

#include "mul_fp16_spatz.h"
#include "mul_fp16_spatz_params.h"
#include "mul_bcast_fp16_spatz_params.h"
#include "mul_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "mul_fp16_spatz"
#define KERNEL_NAME_BCAST "mul_bcast_fp16_spatz"

static int alloc_l1(void **params, uint32_t total_elems)
{
    volatile mul_fp16_spatz_params_t *mul_params;
    uint32_t shard_A;
    uint32_t shard_B;
    uint32_t shard_C;

    size_t elem_start;
    size_t elem_end;
    size_t elem_len;
    size_t elems;
    size_t left;

    elems = total_elems / NUM_HARTS;
    left = total_elems % NUM_HARTS;

    elem_start = HID * elems + (HID < left ? HID : left);
    elem_end = elem_start + elems + (HID < left ? 1 : 0);
    elem_len = elem_end - elem_start;

    l1_alloc_init();

    mul_params = l1_alloc(sizeof(mul_fp16_spatz_params_t));
    if (!mul_params)
        return ENOMEM;

    shard_A = l1_alloc(elem_len * sizeof(float16));
    if (!shard_A)
        return ENOMEM;

    shard_B = l1_alloc(elem_len * sizeof(float16));
    if (!shard_B)
        return ENOMEM;

    shard_C = l1_alloc(elem_len * sizeof(float16));
    if (!shard_C)
        return ENOMEM;

    mul_params->shard_A = shard_A;
    mul_params->shard_B = shard_B;
    mul_params->shard_C = shard_C;
    mul_params->elem_start = elem_start;
    mul_params->elem_len = elem_len;
    mul_params->total_elems = total_elems;

    *params = (void *) mul_params;

    return 0;
}

/*
 * The shard is a contiguous element range of both inputs, so one transfer each. No
 * shard_C prefill: the task writes every element of the shard.
 */
static int init_input_params(void *params, kdma_t *d, const float16 *A, const float16 *B)
{
    volatile mul_fp16_spatz_params_t *mul_params;
    uint32_t bytes;

    mul_params = (volatile mul_fp16_spatz_params_t *) params;
    bytes = mul_params->elem_len * sizeof(float16);

    kdma_in(d, (uintptr_t)(A + mul_params->elem_start), mul_params->shard_A, bytes);
    kdma_in(d, (uintptr_t)(B + mul_params->elem_start), mul_params->shard_B, bytes);

    return 0;
}

static int offload_spatz_task(kdma_t *d, uint32_t task, void *params)
{
    int ret;

    spatz_run_task_with_params(task, (uint32_t)params);

    ret = eu_spatz_wait(&d->eu, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n", HID, KERNEL_NAME, ret);
        return ret;
    }

    return spatz_get_exit_code();
}

static int store_result(void *params, kdma_t *d, float16 *C)
{
    volatile mul_fp16_spatz_params_t *mul_params;

    mul_params = (volatile mul_fp16_spatz_params_t *) params;

    kdma_out(d,
             (uintptr_t)(C + mul_params->elem_start),
             mul_params->shard_C,
             mul_params->elem_len * sizeof(float16));

    return 0;
}

void MAGIA_mul_fp16_spatz(const float16 *A, const float16 *B, float16 *C, uint32_t total_elems)
{
    int ret;
    volatile mul_fp16_spatz_params_t *params;
    kdma_t d;

    ret = alloc_l1((void **)&params, total_elems);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    if (params->elem_len == 0)
        return;

    kdma_open(&d);

    ret = init_input_params((void *)params, &d, A, B);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = offload_spatz_task(&d, MUL_FP16_SPATZ_TASK, (void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result((void *)params, &d, C);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

/*
 * Broadcast multiply, A = [rows, row_len]:
 *
 *   MUL_BCAST_ROW     Y[r][i] = A[r][i] * B[i]   - B is one row_len row, staged whole and
 *                                                  identical on every tile
 *   MUL_BCAST_SCALAR  Y[r][i] = A[r][i] * B[r]   - B is one scalar per row, so only this
 *                                                  tile's rows of it are staged
 *
 * The mesh splits `rows`, which keeps every shard of A and Y a contiguous run.
 */
void MAGIA_mul_bcast_fp16_spatz(const float16 *A,
                                const float16 *B,
                                float16 *Y,
                                uint32_t rows,
                                uint32_t row_len,
                                uint32_t mode)
{
    volatile mul_bcast_fp16_spatz_params_t *params;
    uintptr_t shard_A;
    uintptr_t shard_B;
    uintptr_t shard_Y;
    uint32_t r_start;
    uint32_t r_len;
    uint32_t b_len;
    uint32_t bytes;
    kdma_t d;
    int ret;

    {
        uint32_t shard = rows / NUM_HARTS;
        uint32_t left  = rows % NUM_HARTS;

        r_start = HID * shard + (HID < left ? HID : left);
        r_len   = shard + (HID < left ? 1 : 0);
    }

    if (r_len == 0)
        return;

    b_len = (mode == MUL_BCAST_SCALAR) ? r_len : row_len;

    l1_alloc_init();

    params = l1_alloc(sizeof(mul_bcast_fp16_spatz_params_t));
    shard_A = (uintptr_t)l1_alloc((size_t)r_len * row_len * sizeof(float16));
    shard_B = (uintptr_t)l1_alloc((size_t)b_len * sizeof(float16));
    shard_Y = (uintptr_t)l1_alloc((size_t)r_len * row_len * sizeof(float16));

    if (!params || !shard_A || !shard_B || !shard_Y) {
        printf("[CV32 (%d)] [%s] L1 allocation failed\n", HID, KERNEL_NAME_BCAST);
        return;
    }

    params->shard_A = shard_A;
    params->shard_B = shard_B;
    params->shard_Y = shard_Y;
    params->rows    = r_len;
    params->row_len = row_len;
    params->mode    = mode;

    kdma_open(&d);

    bytes = r_len * row_len * sizeof(float16);
    kdma_in(&d, (uintptr_t)(A + (uintptr_t)r_start * row_len), shard_A, bytes);
    kdma_in(&d,
            (uintptr_t)(B + (mode == MUL_BCAST_SCALAR ? r_start : 0)),
            shard_B,
            b_len * sizeof(float16));

    ret = offload_spatz_task(&d, MUL_BCAST_FP16_SPATZ_TASK, (void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n", HID, KERNEL_NAME_BCAST, ret);
        return;
    }

    kdma_out(&d, (uintptr_t)(Y + (uintptr_t)r_start * row_len), shard_Y, bytes);
}
