#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "silu_fp16_spatz.h"
#include "silu_fp16_spatz_params.h"
#include "silu_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "silu_fp16_spatz"

static int alloc_l1(void **params, uint32_t total_elems)
{
    volatile silu_fp16_spatz_params_t *silu_params;
    uint32_t shard_X;

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

    silu_params = l1_alloc(sizeof(silu_fp16_spatz_params_t));
    if (!silu_params)
        return ENOMEM;

    /* The task works in place: one shard for X and Y. */
    shard_X = (uint32_t)l1_alloc(elem_len * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    silu_params->shard_X     = shard_X;
    silu_params->shard_Y     = shard_X;
    silu_params->elem_start  = elem_start;
    silu_params->elem_len    = elem_len;
    silu_params->total_elems = total_elems;

    *params = (void *)silu_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X)
{
    volatile silu_fp16_spatz_params_t *silu_params;
    uint32_t elem_start;
    uint32_t elem_len;

    silu_params = (volatile silu_fp16_spatz_params_t *)params;
    elem_start  = silu_params->elem_start;
    elem_len    = silu_params->elem_len;

    if (elem_len == 0)
        return 0;

    /* This tile's slice [elem_start, elem_start+elem_len) is contiguous in X. */
    idma_memcpy_1d(
        0, (uint32_t)(X + elem_start), (uint32_t)silu_params->shard_X, elem_len * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    return 0;
}

static int offload_spatz_task(void *params)
{
    int ret;

    spatz_run_task_with_params(SILU_FP16_SPATZ_TASK, (uint32_t)params);

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

/* Y = SiLU(X) on this tile's Spatz, for operands that are already in this tile's L1: no sharding
   across tiles, no L1 allocation and no L2 transfers. Y may alias X. */
void MAGIA_silu_fp16_spatz_l1(const float16 *X, float16 *Y, uint32_t n)
{
    volatile silu_fp16_spatz_params_t params = {
        .shard_X     = (uintptr_t)X,
        .shard_Y     = (uintptr_t)Y,
        .total_elems = n,
        .elem_start  = 0,
        .elem_len    = n,
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

static int store_result(void *params, float16 *Y)
{
    volatile silu_fp16_spatz_params_t *silu_params;
    uint32_t elem_start;
    uint32_t elem_len;

    silu_params = (volatile silu_fp16_spatz_params_t *)params;
    elem_start  = silu_params->elem_start;
    elem_len    = silu_params->elem_len;

    if (elem_len == 0)
        return 0;

    /* This tile's output slice [elem_start, elem_start+elem_len) is contiguous in L2. */
    idma_memcpy_1d(
        1, (uint32_t)(Y + elem_start), (uint32_t)silu_params->shard_Y, elem_len * sizeof(float16));
    eu_idma_wait_o2a(WFE);

    return 0;
}

void MAGIA_silu_fp16_spatz(const float16 *X, float16 *Y, uint32_t total_elems)
{
    int ret;
    volatile silu_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, total_elems);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, X);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    ret = offload_spatz_task((void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    ret = store_result((void *)params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
