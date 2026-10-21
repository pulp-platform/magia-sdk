#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "reducemax_fp16_spatz.h"
#include "reducemax_fp16_spatz_params.h"
#include "reducemax_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "reducemax_fp16_spatz"

static int alloc_l1(void **params, uint32_t outer_dim, uint32_t reduce_dim, uint32_t inner_dim)
{
    volatile reducemax_fp16_spatz_params_t *rm_params;
    uint32_t shard_X;
    uint32_t shard_Y;

    size_t out_start, out_end, out_len;
    size_t elems, left;

    elems = outer_dim / NUM_HARTS;
    left  = outer_dim % NUM_HARTS;

    out_start = HID * elems + (HID < left ? HID : left);
    out_end   = out_start + elems + (HID < left ? 1 : 0);
    out_len   = out_end - out_start;

    l1_alloc_init();

    rm_params = l1_alloc(sizeof(reducemax_fp16_spatz_params_t));
    if (!rm_params)
        return ENOMEM;

    shard_X = l1_alloc(out_len * reduce_dim * inner_dim * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_Y = l1_alloc(out_len * inner_dim * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    rm_params->shard_X     = shard_X;
    rm_params->shard_Y     = shard_Y;
    rm_params->reduce_dim  = reduce_dim;
    rm_params->inner_dim   = inner_dim;
    rm_params->outer_start = out_start;
    rm_params->outer_len   = out_len;

    *params = (void *)rm_params;
    return 0;
}

static int init_input_params(void *params, const float16 *X)
{
    volatile reducemax_fp16_spatz_params_t *rm_params =
        (volatile reducemax_fp16_spatz_params_t *)params;
    uint32_t src_base;
    uint32_t in_elems;

    if (rm_params->outer_len == 0)
        return 0;

    src_base = rm_params->outer_start * rm_params->reduce_dim * rm_params->inner_dim;
    in_elems = rm_params->outer_len * rm_params->reduce_dim * rm_params->inner_dim;

    /* This tile's outer slice [outer_start, outer_start+outer_len) is contiguous in L2. The
       Spatz task fully writes shard_Y (one max per output), so it is not zeroed here. */
    idma_memcpy_1d(
        0, (uint32_t)(X + src_base), (uint32_t)rm_params->shard_X, in_elems * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    return 0;
}

static int offload_spatz_task(void *params)
{
    int ret;

    spatz_run_task_with_params(REDUCEMAX_FP16_SPATZ_TASK, params);

    ret = eu_spatz_wait(WFE);
    if (ret == 0)
        goto exit;

    ret = spatz_get_exit_code();

exit:
    return ret;
}

/* Y = max of X over reduce_dim on this tile's Spatz, for operands that are already in this
   tile's L1: no sharding across tiles, no L1 allocation and no L2 transfers. */
void MAGIA_reducemax_fp16_spatz_l1(
    const float16 *X, float16 *Y, uint32_t outer_len, uint32_t reduce_dim, uint32_t inner_dim)
{
    volatile reducemax_fp16_spatz_params_t params = {
        .shard_X     = (uintptr_t)X,
        .shard_Y     = (uintptr_t)Y,
        .reduce_dim  = reduce_dim,
        .inner_dim   = inner_dim,
        .outer_start = 0,
        .outer_len   = outer_len,
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
    volatile reducemax_fp16_spatz_params_t *rm_params =
        (volatile reducemax_fp16_spatz_params_t *)params;
    uint32_t dst_base;
    uint32_t out_elems;

    if (rm_params->outer_len == 0)
        return 0;

    dst_base  = rm_params->outer_start * rm_params->inner_dim;
    out_elems = rm_params->outer_len * rm_params->inner_dim;

    /* This tile's output outer slice is contiguous in L2. */
    idma_memcpy_1d(
        1, (uint32_t)(Y + dst_base), (uint32_t)rm_params->shard_Y, out_elems * sizeof(float16));
    eu_idma_wait_o2a(WFE);

    return 0;
}

void MAGIA_reducemax_fp16_spatz(
    const float16 *X, float16 *Y, uint32_t outer_dim, uint32_t reduce_dim, uint32_t inner_dim)
{
    int ret;
    volatile reducemax_fp16_spatz_params_t *params;

    ret = alloc_l1(&params, outer_dim, reduce_dim, inner_dim);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params(params, X);
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

    ret = store_result(params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
