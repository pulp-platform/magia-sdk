#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

#include "add_fp16_spatz.h"
#include "add_fp16_spatz_params.h"
#include "add_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "add_fp16_spatz"

static int allocate_l1(void **params, uint32_t size)
{
    volatile add_fp16_spatz_params_t * add_params;
    uintptr_t shard_A;
    uintptr_t shard_B;
    uintptr_t shard_C;

    size_t shard_start;
    size_t shard_end;
    size_t shard_len;
    size_t elems;
    size_t left;

    elems = size / NUM_HARTS;
    left = size % NUM_HARTS;
    shard_start = HID * elems + (HID < left ? HID : left);
    shard_end = shard_start + elems + (HID < left ? 1 : 0);
    shard_len = shard_end - shard_start;

    l1_alloc_init();

    add_params = l1_alloc(sizeof(add_fp16_spatz_params_t));
    if (!add_params)
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

    add_params->shard_A = shard_A;
    add_params->shard_B = shard_B;
    add_params->shard_C = shard_C;
    add_params->start = shard_start;
    add_params->len = shard_len;
    add_params->end = shard_end;

    *params = (void *) add_params;

    return 0;
}

static int init_input_params(void *params, const float16 *A, const float16 *B)
{
    volatile add_fp16_spatz_params_t *add_params;
    uintptr_t shard_A;
    uintptr_t shard_B;
    uintptr_t shard_C;
    uint32_t start;
    uint32_t len;

    add_params = (volatile add_fp16_spatz_params_t *) params;

    shard_A = add_params->shard_A;
    shard_B = add_params->shard_B;
    shard_C = add_params->shard_C;

    start = add_params->start;
    len = add_params->len;

    for (int i = 0; i < len; i++) {
        uint32_t global_idx = start + i;
        uint32_t offset = i * sizeof(float16);

        mmio_fp16(shard_A + offset) = A[global_idx];
        mmio_fp16(shard_B + offset) = B[global_idx];
        mmio_fp16(shard_C + offset) = 0;
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

    spatz_run_task_with_params(ADD_FP16_SPATZ_TASK, params);

    ret = eu_spatz_wait(&eu_ctrl, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n", HID, KERNEL_NAME, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

static int store_result(void* params, float16 *dst)
{
    volatile add_fp16_spatz_params_t *add_params;
    uint32_t shard_C_base;
    uint32_t start;
    uint32_t len;

    add_params = (volatile add_fp16_spatz_params_t *) params;
    shard_C_base = add_params->shard_C;
    start = add_params->start;
    len = add_params->len;

    for (int i = 0; i < len; i++) {
        uint32_t global_idx = start + i;
        uint32_t offset = i * sizeof(float16);
        dst[global_idx] = mmio_fp16(shard_C_base + offset);
    }

    return 0;
}

void MAGIA_add_fp16_spatz(const float16 *A, const float16 *B, float16 *C, uint32_t size)
{
    int ret;
    volatile add_fp16_spatz_params_t *params;

    ret = allocate_l1(&params, size);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params(params, A, B);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = offload_spatz_task(params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result(params, C);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
