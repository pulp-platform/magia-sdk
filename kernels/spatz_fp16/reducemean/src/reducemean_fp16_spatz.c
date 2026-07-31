#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

#include "reducemean_fp16_spatz.h"
#include "reducemean_fp16_spatz_params.h"
#include "reducemean_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "reducemean_fp16_spatz"

static int alloc_l1(void **params, uint32_t outer_dim, uint32_t reduce_dim, uint32_t inner_dim)
{
    volatile reducemean_fp16_spatz_params_t *rm_params;
    uint32_t shard_X;
    uint32_t shard_Y;

    size_t out_start, out_end, out_len;
    size_t elems, left;

    elems = outer_dim / NUM_HARTS;
    left = outer_dim % NUM_HARTS;

    out_start = HID * elems + (HID < left ? HID : left);
    out_end = out_start + elems + (HID < left ? 1 : 0);
    out_len = out_end - out_start;

    l1_alloc_init();

    rm_params = l1_alloc(sizeof(reducemean_fp16_spatz_params_t));
    if (!rm_params) return ENOMEM;

    shard_X = l1_alloc(out_len * reduce_dim * inner_dim * sizeof(float16));
    if (!shard_X) return ENOMEM;

    shard_Y = l1_alloc(out_len * inner_dim * sizeof(float16));
    if (!shard_Y) return ENOMEM;

    rm_params->shard_X = shard_X;
    rm_params->shard_Y = shard_Y;
    rm_params->reduce_dim = reduce_dim;
    rm_params->inner_dim = inner_dim;
    rm_params->outer_start = out_start;
    rm_params->outer_len = out_len;

    *params = (void *) rm_params;
    return 0;
}

static int init_input_params(void *params, const float16 *X)
{
    volatile reducemean_fp16_spatz_params_t *rm_params = (volatile reducemean_fp16_spatz_params_t *) params;

    uint32_t src_base = rm_params->outer_start * rm_params->reduce_dim * rm_params->inner_dim;
    uint32_t total_input_bytes = rm_params->outer_len * rm_params->reduce_dim * rm_params->inner_dim * sizeof(float16);

    for (uint32_t i = 0; i < (total_input_bytes / sizeof(float16)); i++) {
        mmio_fp16(rm_params->shard_X + i * sizeof(float16)) = X[src_base + i];
    }

    uint32_t total_output_elems = rm_params->outer_len * rm_params->inner_dim;
    for (uint32_t i = 0; i < total_output_elems; i++) {
        mmio_fp16(rm_params->shard_Y + i * sizeof(float16)) = 0;
    }

    return 0;
}

static int offload_spatz_task(void *params)
{
    eu_controller_t eu_ctrl;
    eu_config_t eu_cfg;
    int ret;

    eu_cfg.hartid = HID;
    eu_ctrl.base = NULL;
    eu_ctrl.cfg = &eu_cfg;
    eu_ctrl.api = &eu_api;

    spatz_run_task_with_params(REDUCEMEAN_FP16_SPATZ_TASK, params);

    ret = eu_spatz_wait(&eu_ctrl, WFE);
    if (ret == 0) goto exit;

    ret = spatz_get_exit_code();

exit:
    return ret;
}

static int store_result(void *params, float16 *Y)
{
    volatile reducemean_fp16_spatz_params_t *rm_params = (volatile reducemean_fp16_spatz_params_t *) params;

    uint32_t dst_base = rm_params->outer_start * rm_params->inner_dim;
    uint32_t total_output_elems = rm_params->outer_len * rm_params->inner_dim;

    for (uint32_t i = 0; i < total_output_elems; i++) {
        Y[dst_base + i] = mmio_fp16(rm_params->shard_Y + i * sizeof(float16));
    }

    return 0;
}

void MAGIA_reducemean_fp16_spatz(const float16 *X, float16 *Y, uint32_t outer_dim, uint32_t reduce_dim, uint32_t inner_dim)
{
    int ret;
    volatile reducemean_fp16_spatz_params_t *params;

    ret = alloc_l1(&params, outer_dim, reduce_dim, inner_dim);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params(params, X);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = offload_spatz_task(params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result(params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
