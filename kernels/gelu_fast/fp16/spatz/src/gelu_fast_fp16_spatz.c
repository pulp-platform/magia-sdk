#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "gelu_fast_fp16_spatz.h"
#include "gelu_fast_fp16_spatz_params.h"
#include "gelu_fast_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "gelu_fast_fp16_spatz"

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
    volatile gelu_fast_fp16_spatz_params_t *gf_params;
    uintptr_t shard_X;
    uint32_t elem_start;
    uint32_t elem_len;

    tile_elems(len, &elem_start, &elem_len);

    l1_alloc_init();

    gf_params = l1_alloc(sizeof(gelu_fast_fp16_spatz_params_t));
    if (!gf_params)
        return ENOMEM;

    /* The task writes Y over X: one shard for X and Y. */
    shard_X = (uintptr_t)l1_alloc(elem_len * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    gf_params->shard_X = shard_X;
    gf_params->shard_Y = shard_X;
    gf_params->len     = elem_len;

    *params = (void *)gf_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X, uint32_t len)
{
    volatile gelu_fast_fp16_spatz_params_t *gf_params;
    uint32_t elem_start;
    uint32_t elem_len;

    gf_params = (volatile gelu_fast_fp16_spatz_params_t *)params;
    tile_elems(len, &elem_start, &elem_len);

    if (elem_len == 0)
        return 0;

    idma_memcpy_1d(
        0, (uint32_t)(X + elem_start), (uint32_t)gf_params->shard_X, elem_len * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    return 0;
}

static int offload_spatz_task(void *params)
{
    int ret;

    spatz_run_task_with_params(GELU_FAST_FP16_SPATZ_TASK, (uint32_t)params);

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

void MAGIA_gelu_fast_fp16_spatz_l1(float16 *Y, const float16 *X, uint32_t len)
{
    volatile gelu_fast_fp16_spatz_params_t params;
    int ret;

    /* Field by field: GCC hits an internal error on this 3-word volatile initializer. */
    params.shard_X = (uintptr_t)X;
    params.shard_Y = (uintptr_t)Y;
    params.len     = len;

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
    volatile gelu_fast_fp16_spatz_params_t *gf_params;
    uint32_t elem_start;
    uint32_t elem_len;

    gf_params = (volatile gelu_fast_fp16_spatz_params_t *)params;
    tile_elems(len, &elem_start, &elem_len);

    if (elem_len == 0)
        return 0;

    idma_memcpy_1d(
        1, (uint32_t)(Y + elem_start), (uint32_t)gf_params->shard_Y, elem_len * sizeof(float16));
    eu_idma_wait_o2a(WFE);

    return 0;
}

void MAGIA_gelu_fast_fp16_spatz(float16 *Y, const float16 *X, uint32_t len)
{
    int ret;
    volatile gelu_fast_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, len);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, X, len);
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
