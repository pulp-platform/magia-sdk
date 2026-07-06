#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

#include "mul_fp16_spatz.h"
#include "mul_fp16_spatz_params.h"
#include "mul_fp16_spatz_task_bin.h"

#define HID get_hartid()

static int alloc_l1(void **params, uint32_t total_elems)
{
    volatile mul_fp16_spatz_params_t *mul_params;
    uint32_t shard_A;
    uint32_t shard_B;
    uint32_t shard_C;

    size_t elem_start;
    size_t elem_end;
    size_t elem_len;
    size_t elems;
    size_t left;

    elems = total_elems / NUM_HARTS;
    left = total_elems % NUM_HARTS;

    elem_start = HID * elems + (HID < left ? HID : left);
    elem_end = elem_start + elems + (HID < left ? 1 : 0);
    elem_len = elem_end - elem_start;

    l1_alloc_init();

    mul_params = l1_alloc(sizeof(mul_fp16_spatz_params_t));
    if (!mul_params)
        return ENOMEM;

    shard_A = l1_alloc(elem_len * sizeof(float16));
    if (!shard_A)
        return ENOMEM;

    shard_B = l1_alloc(elem_len * sizeof(float16));
    if (!shard_B)
        return ENOMEM;

    shard_C = l1_alloc(elem_len * sizeof(float16));
    if (!shard_C)
        return ENOMEM;

    mul_params->shard_A = shard_A;
    mul_params->shard_B = shard_B;
    mul_params->shard_C = shard_C;
    mul_params->elem_start = elem_start;
    mul_params->elem_len = elem_len;
    mul_params->total_elems = total_elems;

    *params = (void *) mul_params;

    return 0;
}

static int init_input_params(void *params, const float16 *A, const float16 *B)
{
    volatile mul_fp16_spatz_params_t *mul_params;
    uint32_t shard_A;
    uint32_t shard_B;
    uint32_t shard_C;

    size_t elem_start;
    size_t elem_len;

    mul_params = (volatile mul_fp16_spatz_params_t *) params;

    shard_A = mul_params->shard_A;
    shard_B = mul_params->shard_B;
    shard_C = mul_params->shard_C;
    elem_start = mul_params->elem_start;
    elem_len = mul_params->elem_len;

    for (uint32_t i = 0; i < elem_len; i++) {
        uint32_t global_idx = elem_start + i;
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

    spatz_init(SPATZ_BINARY_START);
    spatz_run_task_with_params(MUL_FP16_SPATZ_TASK, params);

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

static int store_result(void *params, float16 *C)
{
    volatile mul_fp16_spatz_params_t *mul_params;
    uint32_t shard_C_base;
    size_t elem_start;
    size_t elem_len;

    mul_params = (volatile mul_fp16_spatz_params_t *) params;
    shard_C_base = mul_params->shard_C;
    elem_start = mul_params->elem_start;
    elem_len = mul_params->elem_len;

    for (uint32_t i = 0; i < elem_len; i++) {
        uint32_t global_idx = elem_start + i;
        uint32_t offset = i * sizeof(float16);

        C[global_idx] = mmio_fp16(shard_C_base + offset);
    }

    return 0;
}

void MAGIA_mul_fp16_spatz(const float16 *A, const float16 *B, float16 *C, uint32_t total_elems)
{
    int ret;
    volatile mul_fp16_spatz_params_t *params;

    ret = alloc_l1(&params, total_elems);
    if (ret != 0) {
        printf("[CV32 (%d)] L1 allocation failed with error: %d\n", HID, ret);
        return;
    }

    ret = init_input_params(params, A, B);
    if (ret != 0) {
        printf("[CV32 (%d)] Params initialization failed with error: %d\n", HID, ret);
        return;
    }

    ret = offload_spatz_task(params);
    if (ret != 0) {
        printf("[CV32 (%d)] Spatz task offloading failed with error: %d\n", HID, ret);
        return;
    }

    ret = store_result(params, C);
    if (ret != 0) {
        printf("[CV32 (%d)] Result write back failed with error: %d\n", HID, ret);
    }
}
