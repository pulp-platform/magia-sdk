#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "kernel_idma_utils.h"

#include "mul_fp16_spatz.h"
#include "mul_fp16_spatz_params.h"
#include "mul_bcast_fp16_spatz_params.h"
#include "mul_fp16_spatz_task_bin.h"

#define HID         get_hartid()
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
    left  = total_elems % NUM_HARTS;

    elem_start = HID * elems + (HID < left ? HID : left);
    elem_end   = elem_start + elems + (HID < left ? 1 : 0);
    elem_len   = elem_end - elem_start;

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

    mul_params->shard_A     = shard_A;
    mul_params->shard_B     = shard_B;
    mul_params->shard_C     = shard_C;
    mul_params->elem_start  = elem_start;
    mul_params->elem_len    = elem_len;
    mul_params->total_elems = total_elems;

    *params = (void *)mul_params;

    return 0;
}

static int init_input_params(void *params, const float16 *A, const float16 *B)
{
    volatile mul_fp16_spatz_params_t *mul_params;
    uint32_t elem_start;
    uint32_t elem_len;

    mul_params = (volatile mul_fp16_spatz_params_t *)params;
    elem_start = mul_params->elem_start;
    elem_len   = mul_params->elem_len;

    if (elem_len == 0)
        return 0;

    idma_ctrl_init();
    eu_ctrl_init();

    /* This tile's slice [elem_start, elem_start+elem_len) is contiguous in both operands.
       The Spatz task writes every output (C = A * B), so shard_C is not zeroed here. */
    idma_memcpy_1d(
        0, (uint32_t)(A + elem_start), (uint32_t)mul_params->shard_A, elem_len * sizeof(float16));
    eu_idma_wait_a2o(WFE);
    idma_memcpy_1d(
        0, (uint32_t)(B + elem_start), (uint32_t)mul_params->shard_B, elem_len * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    return 0;
}

static int offload_spatz_task(void *params)
{
    int ret;

    eu_ctrl_init();
    spatz_run_task_with_params(MUL_FP16_SPATZ_TASK, params);

    ret = eu_spatz_wait(WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

static int store_result(void *params, float16 *C)
{
    volatile mul_fp16_spatz_params_t *mul_params;
    uint32_t elem_start;
    uint32_t elem_len;

    mul_params = (volatile mul_fp16_spatz_params_t *)params;
    elem_start = mul_params->elem_start;
    elem_len   = mul_params->elem_len;

    if (elem_len == 0)
        return 0;

    idma_ctrl_init();
    eu_ctrl_init();

    /* This tile's output slice [elem_start, elem_start+elem_len) is contiguous in L2. */
    idma_memcpy_1d(
        1, (uint32_t)(C + elem_start), (uint32_t)mul_params->shard_C, elem_len * sizeof(float16));
    eu_idma_wait_o2a(WFE);

    return 0;
}

void MAGIA_mul_fp16_spatz(const float16 *A, const float16 *B, float16 *C, uint32_t total_elems)
{
    int ret;
    volatile mul_fp16_spatz_params_t *params;

    ret = alloc_l1(&params, total_elems);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params(params, A, B);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    ret = offload_spatz_task(params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    ret = store_result(params, C);
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

    params  = l1_alloc(sizeof(mul_bcast_fp16_spatz_params_t));
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

    bytes = r_len * row_len * sizeof(float16);
    idma_memcpy_1d(0, (uint32_t)(A + (uintptr_t)r_start * row_len), (uint32_t)shard_A, bytes);
    eu_idma_wait_a2o(WFE);
    idma_memcpy_1d(0,
                   (uint32_t)(B + (mode == MUL_BCAST_SCALAR ? r_start : 0)),
                   (uint32_t)shard_B,
                   b_len * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    spatz_run_task_with_params(MUL_BCAST_FP16_SPATZ_TASK, params);
    if (eu_spatz_wait(WFE) == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed\n", HID, KERNEL_NAME_BCAST);
        return;
    }
    ret = spatz_get_exit_code();
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n",
               HID,
               KERNEL_NAME_BCAST,
               ret);
        return;
    }

    idma_memcpy_1d(1, (uint32_t)(Y + (uintptr_t)r_start * row_len), (uint32_t)shard_Y, bytes);
    eu_idma_wait_o2a(WFE);
}
