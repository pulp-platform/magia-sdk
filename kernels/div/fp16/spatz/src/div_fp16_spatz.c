#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

#include "div_fp16_spatz.h"
#include "div_fp16_spatz_params.h"
#include "div_fp16_spatz_task_bin.h"

#define HID get_hartid()

static int alloc_l1(void **params, uint32_t size)
{
    volatile div_fp16_spatz_params_t *div_params;
    uintptr_t shard_A;
    uintptr_t shard_B;
    uintptr_t shard_C;

    uint32_t shard;
    uint32_t left;
    uint32_t start;
    uint32_t end;
    uint32_t len;

    shard = size / NUM_HARTS;
    left  = size % NUM_HARTS;
    start = HID * shard + (HID < left ? HID : left);
    end   = start + shard + (HID < left ? 1 : 0);
    len   = end - start;

    l1_alloc_init();

    div_params = l1_alloc(sizeof(div_fp16_spatz_params_t));
    if (!div_params)
        return ENOMEM;

    shard_A = l1_alloc(len * sizeof(float16));
    if (!shard_A)
        return ENOMEM;

    shard_B = l1_alloc(len * sizeof(float16));
    if (!shard_B)
        return ENOMEM;

    shard_C = l1_alloc(len * sizeof(float16));
    if (!shard_C)
        return ENOMEM;

    div_params->shard_A = shard_A;
    div_params->shard_B = shard_B;
    div_params->shard_C = shard_C;
    div_params->start   = start;
    div_params->len     = len;
    div_params->end     = end;

    *params = (void *) div_params;

    return 0;
}

static int init_input_params(void *params, const float16 *A, const float16 *B)
{
    volatile div_fp16_spatz_params_t *div_params;
    uintptr_t shard_A_base;
    uintptr_t shard_B_base;
    uintptr_t shard_C_base;
    uint32_t start;
    uint32_t len;

    div_params = (volatile div_fp16_spatz_params_t *) params;
    shard_A_base = div_params->shard_A;
    shard_B_base = div_params->shard_B;
    shard_C_base = div_params->shard_C;

    start = div_params->start;
    len   = div_params->len;

    for (uint32_t i = 0; i < len; i++) {
        uint32_t global_idx = start + i;
        uint32_t offset = i * sizeof(float16);

        mmio_fp16(shard_A_base + offset) = A[global_idx];
        mmio_fp16(shard_B_base + offset) = B[global_idx];
        mmio_fp16(shard_C_base + offset) = 0;
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

    spatz_init(SPATZ_BINARY_START);
    spatz_run_task_with_params(DIV_FP16_SPATZ_TASK, (uint32_t)params);

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

static int store_result(void *params, float16 *dst)
{
    volatile div_fp16_spatz_params_t *div_params;
    uintptr_t shard_C_base;
    uint32_t start;
    uint32_t len;

    div_params = (volatile div_fp16_spatz_params_t *) params;
    shard_C_base = div_params->shard_C;
    start = div_params->start;
    len = div_params->len;

    for (uint32_t i = 0; i < len; i++) {
        uint32_t global_idx = start + i;
        uint32_t offset = i * sizeof(float16);
        dst[global_idx] = mmio_fp16(shard_C_base + offset);
    }

    return 0;
}

void MAGIA_div_fp16_spatz(const float16 *A, const float16 *B, float16 *C, uint32_t size)
{
    int ret;
    volatile div_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, size);
    if (ret != 0) {
        printf("[CV32 (%d)] L1 allocation failed with error: %d\n", HID, ret);
        return;
    }

    ret = init_input_params((void *)params, A, B);
    if (ret != 0) {
        printf("[CV32 (%d)] Params initialization failed with error: %d\n", HID, ret);
        return;
    }

    ret = offload_spatz_task((void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] Spatz task offloading failed with error: %d\n", HID, ret);
        return;
    }

    ret = store_result((void *)params, C);
    if (ret != 0) {
        printf("[CV32 (%d)] Result write back failed with error: %d\n", HID, ret);
    }
}
