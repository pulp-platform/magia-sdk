#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "kernel_idma_utils.h"

#include "col2im_fp16_spatz.h"
#include "col2im_fp16_spatz_params.h"
#include "col2im_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "col2im_fp16_spatz"

static int allocate_l1(void **params, uint32_t batch, uint32_t channels, uint32_t image_h, uint32_t image_w, uint32_t block_h, uint32_t block_w, uint32_t pad_h, uint32_t pad_w, uint32_t stride_h, uint32_t stride_w, uint32_t dilation_h, uint32_t dilation_w, uint32_t l_len)
{
    volatile col2im_fp16_spatz_params_t *col2im_params;
    uintptr_t shard_X;
    uintptr_t shard_Y;

    uint32_t shard;
    uint32_t left;
    uint32_t c_start;
    uint32_t c_end;
    uint32_t c_len;
    uint32_t input_elems;
    uint32_t output_elems;

    shard = channels / NUM_HARTS;
    left  = channels % NUM_HARTS;

    c_start = HID * shard + (HID < left ? HID : left);
    c_end   = c_start + shard + (HID < left ? 1 : 0);
    c_len   = c_end - c_start;

    l1_alloc_init();

    col2im_params = l1_alloc(sizeof(col2im_fp16_spatz_params_t));
    if (!col2im_params)
        return ENOMEM;

    /* shard_X_size: B * shard_C_len * block_h * block_w * l_len  */
    input_elems = batch * c_len * block_h * block_w * l_len;
    shard_X = l1_alloc(input_elems * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    /* shard_Y_size: B * C_len * image_h * image_w */
    output_elems = batch * c_len * image_h * image_w;
    shard_Y = l1_alloc(output_elems * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    col2im_params->shard_X = shard_X;
    col2im_params->shard_Y = shard_Y;
    col2im_params->total_channels = channels;
    col2im_params->batch = batch;
    col2im_params->dilation_h = dilation_h;
    col2im_params->dilation_w = dilation_w;
    col2im_params->stride_h = stride_h;
    col2im_params->stride_w = stride_w;
    col2im_params->image_h = image_h;
    col2im_params->image_w = image_w;
    col2im_params->block_h = block_h;
    col2im_params->block_w = block_w;
    col2im_params->pad_h = pad_h;
    col2im_params->pad_w = pad_w;
    col2im_params->c_start = c_start;
    col2im_params->c_len = c_len;
    col2im_params->l_len = l_len;


    *params = (void *) col2im_params;

    return 0;
}

static int init_input_params(void *params, const float16 *input)
{
    volatile col2im_fp16_spatz_params_t *col2im_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uintptr_t shard_X;
    uintptr_t shard_Y;
    uint32_t image_h;
    uint32_t image_w;
    uint32_t block_h;
    uint32_t block_w;
    uint32_t c_start;
    uint32_t batch;
    uint32_t c_len;
    uint32_t l_len;
    uint32_t tot_c;
    uint32_t bbl;

    col2im_params = (volatile col2im_fp16_spatz_params_t *) params;

    shard_X = col2im_params->shard_X;
    shard_Y = col2im_params->shard_Y;
    image_h = col2im_params->image_h;
    image_w = col2im_params->image_w;
    block_h = col2im_params->block_h;
    block_w = col2im_params->block_w;
    c_start = col2im_params->c_start;
    batch = col2im_params->batch;
    c_len = col2im_params->c_len;
    l_len = col2im_params->l_len;
    tot_c = col2im_params->total_channels;

    if (c_len == 0)
        return 0;

    bbl = block_h * block_w * l_len;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* Column buffer: for each batch this tile's channels [c_start, c_start+c_len) are a
       contiguous run of c_len*bbl elements. Gather them into the packed L1 shard with a 2D
       transfer -- one run per batch, source stride spanning the full channel count. */
    idma_memcpy_2d(&idma_ctrl, 0, (uint32_t) (input + c_start * bbl), (uint32_t) shard_X, c_len * bbl * sizeof(float16), tot_c * bbl * sizeof(float16), batch);
    eu_idma_wait_a2o(&eu_ctrl, WFE);

    /* col2im scatters with accumulation, so the output shard must start zeroed. */
    for (uint32_t i = 0; i < (batch * c_len * image_h * image_w); i++) {
        mmio_fp16(shard_Y + (i * sizeof(float16))) = 0;
    }

    return 0;
}

static int offload_spatz_task(void *params)
{
    eu_controller_t eu_ctrl;
    int ret;

    eu_ctrl_init(&eu_ctrl);

    spatz_run_task_with_params(COL2IM_FP16_SPATZ_TASK, params);

    ret = eu_spatz_wait(&eu_ctrl, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n", HID, KERNEL_NAME, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

static int store_result(void *params, float16 *output)
{
    volatile col2im_fp16_spatz_params_t *col2im_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uintptr_t shard_Y;
    uint32_t image_h;
    uint32_t image_w;
    uint32_t c_start;
    uint32_t c_len;
    uint32_t batch;
    uint32_t tot_c;
    uint32_t ihw;

    col2im_params = (volatile col2im_fp16_spatz_params_t *) params;

    shard_Y = col2im_params->shard_Y;
    image_h = col2im_params->image_h;
    image_w = col2im_params->image_w;
    c_start = col2im_params->c_start;
    c_len = col2im_params->c_len;
    batch = col2im_params->batch;
    tot_c = col2im_params->total_channels;

    if (c_len == 0)
        return 0;

    ihw = image_h * image_w;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* For each batch this tile's output channels [c_start, c_start+c_len) are a contiguous
       run of c_len*ihw elements in L2. Scatter the packed L1 shard back with a 2D transfer
       -- one run per batch, dest stride spanning the full channel count. */
    idma_memcpy_2d(&idma_ctrl, 1, (uint32_t) (output + c_start * ihw), (uint32_t) shard_Y, c_len * ihw * sizeof(float16), tot_c * ihw * sizeof(float16), batch);
    eu_idma_wait_o2a(&eu_ctrl, WFE);

    return 0;
}

void MAGIA_col2im_fp16_spatz(const float16 *input, float16 *output, uint32_t batch, uint32_t channels, uint32_t image_h, uint32_t image_w, uint32_t block_h, uint32_t block_w, uint32_t pad_h, uint32_t pad_w, uint32_t stride_h, uint32_t stride_w, uint32_t dilation_h, uint32_t dilation_w, uint32_t l_len)
{
    int ret;
    volatile col2im_fp16_spatz_params_t *params;

    ret = allocate_l1(&params, batch, channels, image_h, image_w, block_h, block_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, l_len);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params(params, input);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = offload_spatz_task((void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result((void *)params, output);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
