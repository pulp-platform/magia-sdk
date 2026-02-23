#include <stdint.h>

#include "magia_tile_utils.h"
#include "onnx_clip_mem_layout.h"
#include "onnx_clip_params.h"
#include "onnx_clip_task_bin.h"

#include "compare_utils.h"
#include "event_unit_utils.h"
#include "magia_spatz_utils.h"

static int init_data(void *params)
{
    uint32_t offset;
    volatile onnx_clip_params_t *clip_params;

    clip_params = (volatile onnx_clip_params_t *) params;
    for (int i = 0; i < LEN; i++) {
        offset = i * sizeof(float16);

        mmio_fp16(INPUT_VEC_BASE + offset) = input_vec[i];
        mmio_fp16(EXP_BASE + offset) = expected[i];
        mmio_fp16(RES_BASE + offset) = 0;
    }

    mmio_fp16(MIN_VAL_BASE) = min_val;
    mmio_fp16(MAX_VAL_BASE) = max_val;

    clip_params->addr_input = INPUT_VEC_BASE;
    clip_params->addr_min = MIN_VAL_BASE;
    clip_params->addr_max = MAX_VAL_BASE;
    clip_params->addr_exp = EXP_BASE;
    clip_params->addr_res = RES_BASE;
    clip_params->len = LEN;

    return 0;
}

static int run_spatz_task()
{
    int ret;

    eu_init();
    eu_enable_events(EU_SPATZ_DONE_MASK);

    spatz_init(SPATZ_BINARY_START);
    spatz_run_task_with_params(ONNX_CLIP_TASK, ONNX_CLIP_PARAMS_BASE);

    eu_wait_spatz_wfe(EU_SPATZ_DONE_MASK);

    ret = spatz_get_exit_code();

    spatz_clk_dis();

    return ret;
}

static bool check_result(void *params)
{
    volatile onnx_clip_params_t *clip_params;
    clip_params = (volatile onnx_clip_params_t *) params;
    return vector_compare_fp16_bitwise(clip_params->addr_res, clip_params->addr_exp, clip_params->len);
}

static int run_test()
{
    int ret;
    bool check;
    volatile onnx_clip_params_t *params;

    params = (volatile onnx_clip_params_t *) ONNX_CLIP_PARAMS_BASE;

    ret = init_data(params);
    if (ret != 0) {
        printf("[CV32] Params initialization failed with error: %d\n", ret);
        return ret;
    }

    ret = run_spatz_task();
    if (ret != 0) {
        printf("[CV32] Spatz task FAILED with error: %d\n", ret);
        return ret;
    }

    check = check_result(params);
    if (check) {
        printf("[CV32] Test SUCCESS\n");
    } else {
        printf("[CV32] Test FAILED\n");
        ret = -1;
    }

    return ret;
}

int main(void)
{
    int ret;

    printf("\n##################################### ONNX_CLIP TEST #####################################\n\n");

    ret = run_test();

    printf("\n##########################################################################################\n\n");

    return ret;
}
