#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "swiglu_fast_fp16_spatz.h"
#include "swiglu_fast_fp16_spatz_params.h"
#include "swiglu_fast_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "swiglu_fast_fp16_spatz"

/* This tile's elements [*elem_start, *elem_start + *elem_len) out of len. */
static void tile_elems(uint32_t len, uint32_t *elem_start, uint32_t *elem_len)
{
    uint32_t elems = len / NUM_HARTS;
    uint32_t left  = len % NUM_HARTS;

    *elem_start = HID * elems + (HID < left ? HID : left);
    *elem_len   = elems + (HID < left ? 1 : 0);
}

static int alloc_l1(void **params, uint32_t len)
{
    volatile swiglu_fast_fp16_spatz_params_t *sf_params;
    uintptr_t shard_G;
    uintptr_t shard_U;
    uint32_t elem_start;
    uint32_t elem_len;

    tile_elems(len, &elem_start, &elem_len);

    l1_alloc_init();

    sf_params = l1_alloc(sizeof(swiglu_fast_fp16_spatz_params_t));
    if (!sf_params)
        return ENOMEM;

    /* The task writes Y over G: one shard for G and Y. */
    shard_G = (uintptr_t)l1_alloc(elem_len * sizeof(float16));
    if (!shard_G)
        return ENOMEM;

    shard_U = (uintptr_t)l1_alloc(elem_len * sizeof(float16));
    if (!shard_U)
        return ENOMEM;

    sf_params->shard_G = shard_G;
    sf_params->shard_U = shard_U;
    sf_params->shard_Y = shard_G;
    sf_params->len     = elem_len;

    *params = (void *)sf_params;

    return 0;
}

static int init_input_params(void *params, const float16 *G, const float16 *U, uint32_t len)
{
    volatile swiglu_fast_fp16_spatz_params_t *sf_params;
    uint32_t elem_start;
    uint32_t elem_len;

    sf_params = (volatile swiglu_fast_fp16_spatz_params_t *)params;
    tile_elems(len, &elem_start, &elem_len);

    if (elem_len == 0)
        return 0;

    idma_memcpy_1d(
        0, (uint32_t)(G + elem_start), (uint32_t)sf_params->shard_G, elem_len * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    idma_memcpy_1d(
        0, (uint32_t)(U + elem_start), (uint32_t)sf_params->shard_U, elem_len * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    return 0;
}

static int offload_spatz_task(void *params)
{
    int ret;

    spatz_run_task_with_params(SWIGLU_FAST_FP16_SPATZ_TASK, (uint32_t)params);

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

void MAGIA_swiglu_fast_fp16_spatz_l1(float16 *Y, const float16 *G, const float16 *U, uint32_t len)
{
    volatile swiglu_fast_fp16_spatz_params_t params = {
        .shard_G = (uintptr_t)G,
        .shard_U = (uintptr_t)U,
        .shard_Y = (uintptr_t)Y,
        .len     = len,
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

static int store_result(void *params, float16 *Y, uint32_t len)
{
    volatile swiglu_fast_fp16_spatz_params_t *sf_params;
    uint32_t elem_start;
    uint32_t elem_len;

    sf_params = (volatile swiglu_fast_fp16_spatz_params_t *)params;
    tile_elems(len, &elem_start, &elem_len);

    if (elem_len == 0)
        return 0;

    idma_memcpy_1d(
        1, (uint32_t)(Y + elem_start), (uint32_t)sf_params->shard_Y, elem_len * sizeof(float16));
    eu_idma_wait_o2a(WFE);

    return 0;
}

void MAGIA_swiglu_fast_fp16_spatz(float16 *Y, const float16 *G, const float16 *U, uint32_t len)
{
    int ret;
    volatile swiglu_fast_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, len);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, G, U, len);
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

    ret = store_result((void *)params, Y, len);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
