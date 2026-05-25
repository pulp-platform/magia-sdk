#include <stdint.h>

#include "eventunit.h"
#include "tile.h"

#include "globalaveragepool_fp16_spatz.h"
#include "globalaveragepool_fp16_spatz_mem_layout.h"
#include "globalaveragepool_fp16_spatz_params.h"
#include "globalaveragepool_fp16_spatz_task_bin.h"

#define HID get_hartid()

static int init_input_params(void *params, const float16 *X)
{
    volatile globalaveragepool_fp16_spatz_params_t *gap_params;
    uint32_t shard;
    uint32_t left;
    uint32_t start;
    uint32_t end;
    uint32_t len;
    uint32_t hw_len;
    uint32_t local_idx;

    gap_params = (volatile globalaveragepool_fp16_spatz_params_t *) params;

    shard = CHANNEL_NUM / NUM_HARTS;
    left  = CHANNEL_NUM % NUM_HARTS;

    start = HID * shard + (HID < left ? HID : left);
    end = start + shard + (HID < left ? 1 : 0);
    len = end - start;
    hw_len = CHANNEL_LEN;

    local_idx = 0;
    for (int inst = start; inst < end; inst++) {
        uint32_t global_base = inst * hw_len;

        for (int i = 0; i < hw_len; i++) {
            uint32_t global_idx = global_base + i;
            uint32_t offset = local_idx * sizeof(float16);

            mmio_fp16(SHARD_X_BASE + offset) = X[global_idx];
            local_idx++;
        }
    }

    for (int i = 0; i < len; i++) {
        uint32_t offset = i * sizeof(float16);
        mmio_fp16(SHARD_Y_BASE + offset) = 0;
    }

    gap_params->shard_X = SHARD_X_BASE;
    gap_params->shard_Y = SHARD_Y_BASE;
    gap_params->hw_len = hw_len;
    gap_params->start = start;
    gap_params->len = len;

    return 0;
}

static int offload_spatz_task()
{
    eu_controller_t eu_ctrl;
    eu_config_t eu_cfg;
    int ret;

    eu_cfg.hartid = HID;
    eu_ctrl.base = NULL;
    eu_ctrl.cfg = &eu_cfg;
    eu_ctrl.api = &eu_api;

    spatz_init(SPATZ_BINARY_START);
    spatz_run_task_with_params(GLOBALAVERAGEPOOL_FP16_SPATZ_TASK, GLOBALAVERAGEPOOL_FP16_SPATZ_PARAMS_BASE);

    ret = eu_spatz_wait(&eu_ctrl, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] Wait on Spatz task completion failed with error: %d\n", HID, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();
    spatz_clk_dis();

exit:
    return ret;
}

static int store_result(void *params, float16 *Y)
{
    volatile globalaveragepool_fp16_spatz_params_t *gap_params;
    uint32_t shard_Y_base;

    gap_params = (volatile globalaveragepool_fp16_spatz_params_t *) params;
    shard_Y_base = gap_params->shard_Y;

    for (int i = 0; i < gap_params->len; i++) {
        uint32_t global_idx = gap_params->start + i;
        uint32_t offset = i * sizeof(float16);

        Y[global_idx] = mmio_fp16(shard_Y_base + offset);
    }

    return 0;
}

void MAGIA_globalaveragepool_fp16_spatz(const float16 *X, float16 *Y)
{
    int ret;
    volatile globalaveragepool_fp16_spatz_params_t *params;

    params = (volatile globalaveragepool_fp16_spatz_params_t *) GLOBALAVERAGEPOOL_FP16_SPATZ_PARAMS_BASE;

    ret = init_input_params(params, X);
    if (ret != 0) {
        printf("[CV32 (%d)] Params initialization failed with error: %d\n", HID, ret);
        return;
    }

    ret = offload_spatz_task();
    if (ret != 0) {
        printf("[CV32 (%d)] Spatz task offloading failed with error: %d\n", HID, ret);
        return;
    }

    ret = store_result(params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] Result write back failed with error: %d\n", HID, ret);
    }
}
