#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

#include "reducesum_fp16_spatz.h"
#include "reducesum_fp16_spatz_params.h"
#include "reducesum_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "reducesum_fp16_spatz"

static int alloc_l1(void **params, uint32_t outer_dim, uint32_t reduce_dim, uint32_t inner_dim)
{
    volatile reducesum_fp16_spatz_params_t *rs_params;
    uintptr_t shard_X;
    uintptr_t shard_Y;

    size_t out_start, out_end, out_len;
    size_t elems, left;

    elems = outer_dim / NUM_HARTS;
    left = outer_dim % NUM_HARTS;

    out_start = HID * elems + (HID < left ? HID : left);
    out_end = out_start + elems + (HID < left ? 1 : 0);
    out_len = out_end - out_start;

    l1_alloc_init();

    rs_params = l1_alloc(sizeof(reducesum_fp16_spatz_params_t));
    if (!rs_params) return ENOMEM;

    shard_X = (uintptr_t) l1_alloc(out_len * reduce_dim * inner_dim * sizeof(float16));
    if (!shard_X) return ENOMEM;

    shard_Y = (uintptr_t) l1_alloc(out_len * inner_dim * sizeof(float16));
    if (!shard_Y) return ENOMEM;

    rs_params->shard_X = shard_X;
    rs_params->shard_Y = shard_Y;
    rs_params->reduce_dim = reduce_dim;
    rs_params->inner_dim = inner_dim;
    rs_params->outer_start = out_start;
    rs_params->outer_len = out_len;

    *params = (void *) rs_params;
    return 0;
}

static int init_input_params(void *params, const float16 *X)
{
    volatile reducesum_fp16_spatz_params_t *rs_params = (volatile reducesum_fp16_spatz_params_t *) params;

    uint32_t src_base = rs_params->outer_start * rs_params->reduce_dim * rs_params->inner_dim;
    uint32_t total_input_elems = rs_params->outer_len * rs_params->reduce_dim * rs_params->inner_dim;

    for (uint32_t i = 0; i < total_input_elems; i++) {
        mmio_fp16(rs_params->shard_X + i * sizeof(float16)) = X[src_base + i];
    }

    uint32_t total_output_elems = rs_params->outer_len * rs_params->inner_dim;
    for (uint32_t i = 0; i < total_output_elems; i++) {
        mmio_fp16(rs_params->shard_Y + i * sizeof(float16)) = 0;
    }

    return 0;
}

static int offload_spatz_task(void *params)
{
    eu_controller_t eu_ctrl;
    eu_config_t eu_cfg;
    int ret;

    eu_cfg.hartid = HID;
    eu_ctrl.base = 0;      /* uint32_t MMIO base, unused by the Spatz events */
    eu_ctrl.cfg = &eu_cfg;
    eu_ctrl.api = &eu_api;

    spatz_run_task_with_params(REDUCESUM_FP16_SPATZ_TASK, (uint32_t)(uintptr_t) params);

    /* eu_spatz_wait() returns non-zero on completion, 0 on timeout */
    ret = eu_spatz_wait(&eu_ctrl, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion timed out\n", HID, KERNEL_NAME);
        return ETIMEDOUT;
    }

    return spatz_get_exit_code();
}

static int store_result(void *params, float16 *Y)
{
    volatile reducesum_fp16_spatz_params_t *rs_params = (volatile reducesum_fp16_spatz_params_t *) params;

    uint32_t dst_base = rs_params->outer_start * rs_params->inner_dim;
    uint32_t total_output_elems = rs_params->outer_len * rs_params->inner_dim;

    for (uint32_t i = 0; i < total_output_elems; i++) {
        Y[dst_base + i] = mmio_fp16(rs_params->shard_Y + i * sizeof(float16));
    }

    return 0;
}

void MAGIA_reducesum_fp16_spatz(const float16 *X, float16 *Y, uint32_t outer_dim, uint32_t reduce_dim, uint32_t inner_dim)
{
    int ret;
    volatile reducesum_fp16_spatz_params_t *params;

    ret = alloc_l1((void **) &params, outer_dim, reduce_dim, inner_dim);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *) params, X);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = offload_spatz_task((void *) params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result((void *) params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
