#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "swiglu_fp16_spatz.h"
#include "swiglu_fp16_spatz_params.h"
#include "swiglu_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "swiglu_fp16_spatz"

static int alloc_l1(void **params, uint32_t total_elems)
{
    volatile swiglu_fp16_spatz_params_t *sw_params;
    uint32_t shard_G;
    uint32_t shard_U;

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

    sw_params = l1_alloc(sizeof(swiglu_fp16_spatz_params_t));
    if (!sw_params)
        return ENOMEM;

    /* The task writes Y over G: one shard for G and Y. */
    shard_G = (uint32_t)l1_alloc(elem_len * sizeof(float16));
    if (!shard_G)
        return ENOMEM;

    shard_U = (uint32_t)l1_alloc(elem_len * sizeof(float16));
    if (!shard_U)
        return ENOMEM;

    sw_params->shard_G     = shard_G;
    sw_params->shard_U     = shard_U;
    sw_params->shard_Y     = shard_G;
    sw_params->elem_start  = elem_start;
    sw_params->elem_len    = elem_len;
    sw_params->total_elems = total_elems;

    *params = (void *)sw_params;

    return 0;
}

static int init_input_params(void *params, const float16 *G, const float16 *U)
{
    volatile swiglu_fp16_spatz_params_t *sw_params;
    uint32_t elem_start;
    uint32_t elem_len;

    sw_params = (volatile swiglu_fp16_spatz_params_t *)params;
    elem_start  = sw_params->elem_start;
    elem_len    = sw_params->elem_len;

    if (elem_len == 0)
        return 0;

    /* This tile's slice [elem_start, elem_start+elem_len) is contiguous in G and U. */
    idma_memcpy_1d(
        0, (uint32_t)(G + elem_start), (uint32_t)sw_params->shard_G, elem_len * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    idma_memcpy_1d(
        0, (uint32_t)(U + elem_start), (uint32_t)sw_params->shard_U, elem_len * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    return 0;
}

static int offload_spatz_task(void *params)
{
    int ret;

    spatz_run_task_with_params(SWIGLU_FP16_SPATZ_TASK, (uint32_t)params);

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

/* Y = SiLU(G) * U on this tile's Spatz, for operands that are already in this tile's L1: no
   sharding across tiles, no L1 allocation and no L2 transfers. Y may alias G or U. */
void MAGIA_swiglu_fp16_spatz_l1(const float16 *G, const float16 *U, float16 *Y, uint32_t n)
{
    volatile swiglu_fp16_spatz_params_t params = {
        .shard_G     = (uintptr_t)G,
        .shard_U     = (uintptr_t)U,
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
    volatile swiglu_fp16_spatz_params_t *sw_params;
    uint32_t elem_start;
    uint32_t elem_len;

    sw_params = (volatile swiglu_fp16_spatz_params_t *)params;
    elem_start  = sw_params->elem_start;
    elem_len    = sw_params->elem_len;

    if (elem_len == 0)
        return 0;

    /* This tile's output slice [elem_start, elem_start+elem_len) is contiguous in L2. */
    idma_memcpy_1d(
        1, (uint32_t)(Y + elem_start), (uint32_t)sw_params->shard_Y, elem_len * sizeof(float16));
    eu_idma_wait_o2a(WFE);

    return 0;
}

void MAGIA_swiglu_fp16_spatz(const float16 *G, const float16 *U, float16 *Y, uint32_t total_elems)
{
    int ret;
    volatile swiglu_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, total_elems);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, G, U);
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
