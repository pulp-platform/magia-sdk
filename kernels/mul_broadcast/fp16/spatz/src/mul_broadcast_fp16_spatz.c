#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "mul_broadcast_fp16_spatz.h"
#include "mul_broadcast_fp16_spatz_params.h"
#include "mul_broadcast_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "mul_broadcast_fp16_spatz"

static int alloc_l1(void **params, const float16 *B, uint32_t total_elems)
{
    volatile mul_broadcast_fp16_spatz_params_t *mul_params;
    uint32_t shard_A;
    uint32_t shard_C;
    uint32_t scalar;

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

    mul_params = l1_alloc(sizeof(mul_broadcast_fp16_spatz_params_t));
    if (!mul_params)
        return ENOMEM;

    shard_A = l1_alloc(elem_len * sizeof(float16));
    if (!shard_A)
        return ENOMEM;

    shard_C = l1_alloc(elem_len * sizeof(float16));
    if (!shard_C)
        return ENOMEM;

    scalar = l1_alloc(sizeof(float16));
    if (!scalar)
        return ENOMEM;

    /* The scalar is a single element: the CV32 copies it to L1 itself, no iDMA needed. */
    mmio_fp16(scalar) = B[0];

    mul_params->shard_A     = shard_A;
    mul_params->shard_C     = shard_C;
    mul_params->scalar      = scalar;
    mul_params->elem_start  = elem_start;
    mul_params->elem_len    = elem_len;
    mul_params->total_elems = total_elems;

    *params = (void *)mul_params;

    return 0;
}

static int init_input_params(void *params, const float16 *A)
{
    volatile mul_broadcast_fp16_spatz_params_t *mul_params;
    uint32_t elem_start;
    uint32_t elem_len;

    mul_params = (volatile mul_broadcast_fp16_spatz_params_t *)params;
    elem_start = mul_params->elem_start;
    elem_len   = mul_params->elem_len;

    if (elem_len == 0)
        return 0;

    /* This tile's slice [elem_start, elem_start+elem_len) is contiguous in A.
       The Spatz task writes every output (C = A * s), so shard_C is not zeroed here. */
    idma_memcpy_1d(
        0, (uint32_t)(A + elem_start), (uint32_t)mul_params->shard_A, elem_len * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    return 0;
}

static int offload_spatz_task(void *params)
{
    int ret;

    spatz_run_task_with_params(MUL_BROADCAST_FP16_SPATZ_TASK, params);

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

/* C = A * B[0] on this tile's Spatz, for operands that are already in this tile's L1:
   no sharding across tiles, no L1 allocation and no L2 transfers. C may alias A. */
void MAGIA_mul_broadcast_fp16_spatz_l1(const float16 *A, const float16 *B, float16 *C, uint32_t n)
{
    volatile mul_broadcast_fp16_spatz_params_t params = {
        .shard_A     = (uintptr_t)A,
        .shard_C     = (uintptr_t)C,
        .scalar      = (uintptr_t)B,
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

static int store_result(void *params, float16 *C)
{
    volatile mul_broadcast_fp16_spatz_params_t *mul_params;
    uint32_t elem_start;
    uint32_t elem_len;

    mul_params = (volatile mul_broadcast_fp16_spatz_params_t *)params;
    elem_start = mul_params->elem_start;
    elem_len   = mul_params->elem_len;

    if (elem_len == 0)
        return 0;

    /* This tile's output slice [elem_start, elem_start+elem_len) is contiguous in L2. */
    idma_memcpy_1d(
        1, (uint32_t)(C + elem_start), (uint32_t)mul_params->shard_C, elem_len * sizeof(float16));
    eu_idma_wait_o2a(WFE);

    return 0;
}

void MAGIA_mul_broadcast_fp16_spatz(const float16 *A,
                                    const float16 *B,
                                    float16 *C,
                                    uint32_t total_elems)
{
    int ret;
    volatile mul_broadcast_fp16_spatz_params_t *params;

    ret = alloc_l1(&params, B, total_elems);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params(params, A);
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

    ret = store_result(params, C);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
