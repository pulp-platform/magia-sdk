#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

#include "kernels_dma_utils.h"

#include "add_fp16_spatz.h"
#include "add_fp16_spatz_params.h"
#include "add_bcast_fp16_spatz_params.h"
#include "add_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "add_fp16_spatz"
#define KERNEL_NAME_BCAST "add_bcast_fp16_spatz"

static int allocate_l1(void **params, uint32_t size)
{
    volatile add_fp16_spatz_params_t * add_params;
    uintptr_t shard_A;
    uintptr_t shard_B;
    uintptr_t shard_C;

    size_t shard_start;
    size_t shard_end;
    size_t shard_len;
    size_t elems;
    size_t left;

    elems = size / NUM_HARTS;
    left = size % NUM_HARTS;
    shard_start = HID * elems + (HID < left ? HID : left);
    shard_end = shard_start + elems + (HID < left ? 1 : 0);
    shard_len = shard_end - shard_start;

    l1_alloc_init();

    add_params = l1_alloc(sizeof(add_fp16_spatz_params_t));
    if (!add_params)
        return ENOMEM;

    shard_A = l1_alloc(shard_len * sizeof(float16));
    if (!shard_A)
        return ENOMEM;

    shard_B = l1_alloc(shard_len * sizeof(float16));
    if (!shard_B)
        return ENOMEM;

    shard_C = l1_alloc(shard_len * sizeof(float16));
    if (!shard_C)
        return ENOMEM;

    add_params->shard_A = shard_A;
    add_params->shard_B = shard_B;
    add_params->shard_C = shard_C;
    add_params->start = shard_start;
    add_params->len = shard_len;
    add_params->end = shard_end;

    *params = (void *) add_params;

    return 0;
}

/* This tile's shard is a contiguous range of both operands, so one transfer each brings
 * them in. shard_C is not pre-cleared: the task writes every element of it. */
static int init_input_params(kdma_t *d, void *params, const float16 *A, const float16 *B)
{
    volatile add_fp16_spatz_params_t *add_params;
    uint32_t bytes;

    add_params = (volatile add_fp16_spatz_params_t *) params;
    bytes = add_params->len * sizeof(float16);

    kdma_in(d, (uintptr_t) (A + add_params->start), add_params->shard_A, bytes);
    kdma_in(d, (uintptr_t) (B + add_params->start), add_params->shard_B, bytes);

    return 0;
}

static int offload_spatz_task(kdma_t *d, uint32_t task, void *params)
{
    int ret;

    spatz_run_task_with_params(task, params);

    ret = eu_spatz_wait(&d->eu, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n", HID, KERNEL_NAME, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

static int store_result(kdma_t *d, void* params, float16 *dst)
{
    volatile add_fp16_spatz_params_t *add_params;

    add_params = (volatile add_fp16_spatz_params_t *) params;

    kdma_out(d,
             (uintptr_t) (dst + add_params->start),
             add_params->shard_C,
             add_params->len * sizeof(float16));

    return 0;
}

void MAGIA_add_fp16_spatz(const float16 *A, const float16 *B, float16 *C, uint32_t size)
{
    int ret;
    volatile add_fp16_spatz_params_t *params;
    kdma_t d;

    kdma_open(&d);

    ret = allocate_l1(&params, size);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    /* More tiles than elements: nothing staged, nothing offloaded, nothing written back. */
    if (params->len == 0)
        return;

    ret = init_input_params(&d, (void *)params, A, B);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = offload_spatz_task(&d, ADD_FP16_SPATZ_TASK, (void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result(&d, (void *)params, C);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

/*
 * Broadcast add, A = [rows, row_len]:
 *
 *   ADD_BCAST_ROW     Y[r][i] = A[r][i] + B[i]   - B is one row_len row, staged whole and
 *                                                  identical on every tile
 *   ADD_BCAST_SCALAR  Y[r][i] = A[r][i] + B[r]   - B is one scalar per row, so only this
 *                                                  tile's rows of it are staged
 *
 * The mesh splits `rows`, which keeps every shard of A and Y a contiguous run.
 */
void MAGIA_add_bcast_fp16_spatz(const float16 *A,
                                const float16 *B,
                                float16 *Y,
                                uint32_t rows,
                                uint32_t row_len,
                                uint32_t mode)
{
    volatile add_bcast_fp16_spatz_params_t *params;
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

    b_len = (mode == ADD_BCAST_SCALAR) ? r_len : row_len;

    l1_alloc_init();

    params = l1_alloc(sizeof(add_bcast_fp16_spatz_params_t));
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
            (uintptr_t)(B + (mode == ADD_BCAST_SCALAR ? r_start : 0)),
            shard_B,
            b_len * sizeof(float16));

    ret = offload_spatz_task(&d, ADD_BCAST_FP16_SPATZ_TASK, (void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n", HID, KERNEL_NAME_BCAST, ret);
        return;
    }

    kdma_out(&d, (uintptr_t)(Y + (uintptr_t)r_start * row_len), shard_Y, bytes);
}
