#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "kernel_idma_utils.h"

#include "transpose_fp16_spatz.h"
#include "transpose_fp16_spatz_params.h"
#include "transpose_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "transpose_fp16_spatz"

static int alloc_l1(void **params, uint32_t *in_shape, uint32_t *out_shape, uint32_t rank, uint32_t iterations, uint32_t *out_shard_in_elems, uint32_t *out_shard_out_elems)
{
    volatile transpose_fp16_spatz_params_t *trans_params;

    uintptr_t shard_input;
    uintptr_t shard_output;
    uintptr_t out_shape_l1;
    uintptr_t in_strides_l1;
    uintptr_t perm_l1;
    uintptr_t coord_l1;

    uint32_t shard_in_elems;
    uint32_t shard_out_elems;
    uint32_t iter_start;
    uint32_t iter_end;
    uint32_t iter_len;
    uint32_t shard;
    uint32_t left;
    uint32_t stride;

    shard_in_elems  = 1;
    shard_out_elems = 1;
    for (uint32_t i = 1; i < rank; i++) {
        shard_in_elems  *= in_shape[i];
        shard_out_elems *= out_shape[i];
    }

    shard = iterations / NUM_HARTS;
    left  = iterations % NUM_HARTS;

    iter_start = HID * shard + (HID < left ? HID : left);
    iter_end   = iter_start + shard + (HID < left ? 1 : 0);
    iter_len   = iter_end - iter_start;

    l1_alloc_init();

    trans_params = l1_alloc(sizeof(transpose_fp16_spatz_params_t));
    if (!trans_params)
        return ENOMEM;

    shard_input = l1_alloc(iter_len * shard_in_elems * sizeof(float16));
    if (!shard_input)
        return ENOMEM;

    shard_output = l1_alloc(iter_len * shard_out_elems * sizeof(float16));
    if (!shard_output)
        return ENOMEM;

    out_shape_l1 = l1_alloc(rank * sizeof(uint32_t));
    if (!out_shape_l1)
        return ENOMEM;

    in_strides_l1 = l1_alloc(rank * sizeof(uint32_t));
    if (!in_strides_l1)
        return ENOMEM;

    perm_l1 = l1_alloc(rank * sizeof(uint32_t));
    if (!perm_l1)
        return ENOMEM;

    coord_l1 = l1_alloc(rank * sizeof(uint32_t));
    if (!coord_l1)
        return ENOMEM;

    trans_params->shard_input     = shard_input;
    trans_params->shard_output    = shard_output;
    trans_params->out_shape       = out_shape_l1;
    trans_params->in_strides      = in_strides_l1;
    trans_params->perm            = perm_l1;
    trans_params->coord           = coord_l1;
    trans_params->rank            = rank;
    trans_params->iteration_start = iter_start;
    trans_params->iteration_len   = iter_len;

    for (uint32_t i = 0; i < rank; i++)
        mmio32(out_shape_l1 + i * sizeof(uint32_t)) = out_shape[i];

    stride = 1;
    for (int i = (int) rank - 1; i >= 0; i--) {
        mmio32(in_strides_l1 + i * sizeof(uint32_t)) = stride;
        stride *= in_shape[i];
    }

    *params = (void *) trans_params;
    *out_shard_in_elems = shard_in_elems;
    *out_shard_out_elems = shard_out_elems;

    return 0;
}

static int init_input_params(void *params, const float16 *input, const uint32_t *perm, uint32_t shard_in_elems, uint32_t shard_out_elems)
{
    volatile transpose_fp16_spatz_params_t *trans_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uintptr_t perm_base;
    uint32_t rank;
    uint32_t iter_len;
    uint32_t global_offset;
    uint32_t in_bytes;

    trans_params = (volatile transpose_fp16_spatz_params_t *) params;
    perm_base = trans_params->perm;
    rank      = trans_params->rank;
    iter_len  = trans_params->iteration_len;

    /* Control metadata: small rank-sized arrays, staged scalar. */
    for (uint32_t i = 0; i < rank; i++)
        mmio32(perm_base + i * sizeof(uint32_t)) = perm[i];

    if (iter_len == 0)
        return 0;

    global_offset = trans_params->iteration_start * shard_in_elems;
    in_bytes = iter_len * shard_in_elems * sizeof(float16);

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* perm[0] == 0, so this tile's slice of the outer axis is contiguous in L2. The
       Spatz task performs the (strided) transpose and fully writes shard_output, so
       shard_output is not zeroed here. */
    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t) (input + global_offset), (uint32_t) trans_params->shard_input, in_bytes);
    eu_idma_wait_a2o(&eu_ctrl, WFE);

    return 0;
}

static int offload_spatz_task(void *params)
{
    eu_controller_t eu_ctrl;
    int ret;

    eu_ctrl_init(&eu_ctrl);
    spatz_run_task_with_params(TRANSPOSE_FP16_SPATZ_TASK, (uint32_t)params);

    ret = eu_spatz_wait(&eu_ctrl, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n", HID, KERNEL_NAME, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

static int store_result(void *params, float16 *output, uint32_t shard_out_elems)
{
    volatile transpose_fp16_spatz_params_t *trans_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t iter_len;
    uint32_t global_offset;
    uint32_t out_bytes;

    trans_params = (volatile transpose_fp16_spatz_params_t *) params;
    iter_len = trans_params->iteration_len;

    if (iter_len == 0)
        return 0;

    global_offset = trans_params->iteration_start * shard_out_elems;
    out_bytes = iter_len * shard_out_elems * sizeof(float16);

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* This tile's output slice of the outer axis is contiguous in L2. */
    idma_memcpy_1d(&idma_ctrl, 1, (uint32_t) (output + global_offset), (uint32_t) trans_params->shard_output, out_bytes);
    eu_idma_wait_o2a(&eu_ctrl, WFE);

    return 0;
}

void MAGIA_transpose_fp16_spatz(const float16 *input, float16 *output, uint32_t *perm, uint32_t *in_shape, uint32_t *out_shape, uint32_t rank, uint32_t iterations)
{
    int ret;
    volatile transpose_fp16_spatz_params_t *params;
    uint32_t shard_in_elems;
    uint32_t shard_out_elems;

    ret = alloc_l1((void **)&params, in_shape, out_shape, rank, iterations, &shard_in_elems, &shard_out_elems);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, input, perm, shard_in_elems, shard_out_elems);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = offload_spatz_task((void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result((void *)params, output, shard_out_elems);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
