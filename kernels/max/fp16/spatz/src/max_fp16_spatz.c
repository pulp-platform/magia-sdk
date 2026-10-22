#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "max_fp16_spatz.h"
#include "max_fp16_spatz_params.h"
#include "max_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "max_fp16_spatz"

static int allocate_l1(void **params, uint32_t size)
{
    volatile max_fp16_spatz_params_t *max_params;
    uintptr_t shard_A;
    uintptr_t shard_B;
    uintptr_t shard_C;

    size_t shard_start;
    size_t shard_end;
    size_t shard_len;
    size_t elems;
    size_t left;

    elems       = size / NUM_HARTS;
    left        = size % NUM_HARTS;
    shard_start = HID * elems + (HID < left ? HID : left);
    shard_end   = shard_start + elems + (HID < left ? 1 : 0);
    shard_len   = shard_end - shard_start;

    l1_alloc_init();

    max_params = l1_alloc(sizeof(max_fp16_spatz_params_t));
    if (!max_params)
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

    max_params->shard_A = shard_A;
    max_params->shard_B = shard_B;
    max_params->shard_C = shard_C;
    max_params->start   = shard_start;
    max_params->len     = shard_len;
    max_params->end     = shard_end;

    *params = (void *)max_params;

    return 0;
}

static int init_input_params(void *params, const float16 *A, const float16 *B)
{
    volatile max_fp16_spatz_params_t *max_params;
    uint32_t start;
    uint32_t len;

    max_params = (volatile max_fp16_spatz_params_t *)params;
    start      = max_params->start;
    len        = max_params->len;

    if (len == 0)
        return 0;

    /* This tile's slice [start, start+len) is contiguous in both operands. The Spatz task
       writes every output (C = max(A, B)), so shard_C is not zeroed here. */
    idma_memcpy_1d(0, (uint32_t)(A + start), (uint32_t)max_params->shard_A, len * sizeof(float16));
    eu_idma_wait_a2o(WFE);
    idma_memcpy_1d(0, (uint32_t)(B + start), (uint32_t)max_params->shard_B, len * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    return 0;
}

static int offload_spatz_task(void *params)
{
    int ret;

    spatz_run_task_with_params(MAX_FP16_SPATZ_TASK, params);

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

/* C = max(A, B) on this tile's Spatz, for operands that are already in this tile's L1: no
   sharding across tiles, no L1 allocation and no L2 transfers. C may alias A or B. */
void MAGIA_max_fp16_spatz_l1(const float16 *A, const float16 *B, float16 *C, uint32_t size)
{
    volatile max_fp16_spatz_params_t params = {
        .shard_A = (uintptr_t)A,
        .shard_B = (uintptr_t)B,
        .shard_C = (uintptr_t)C,
        .start   = 0,
        .len     = size,
        .end     = size,
    };
    int ret;

    ret = offload_spatz_task((void *)&params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
    }
}

static int store_result(void *params, float16 *dst)
{
    volatile max_fp16_spatz_params_t *max_params;
    uint32_t start;
    uint32_t len;

    max_params = (volatile max_fp16_spatz_params_t *)params;
    start      = max_params->start;
    len        = max_params->len;

    if (len == 0)
        return 0;

    /* This tile's output slice [start, start+len) is contiguous in L2. */
    idma_memcpy_1d(
        1, (uint32_t)(dst + start), (uint32_t)max_params->shard_C, len * sizeof(float16));
    eu_idma_wait_o2a(WFE);

    return 0;
}

void MAGIA_max_fp16_spatz(const float16 *A, const float16 *B, float16 *C, uint32_t size)
{
    int ret;
    volatile max_fp16_spatz_params_t *params;

    ret = allocate_l1(&params, size);
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
