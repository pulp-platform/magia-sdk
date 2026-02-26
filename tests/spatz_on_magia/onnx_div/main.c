#include <stdint.h>

#include "compare_utils.h"
#include "data.h"
#include "event_unit_utils.h"
#include "onnx_div_mem_layout.h"
#include "onnx_div_params.h"
#include "onnx_div_task_bin.h"

#include "magia_tile_utils.h"
#include "magia_spatz_utils.h"

static int init_data(void *params)
{
    uint32_t offset;
    volatile onnx_div_params_t *div_params;

    div_params = (volatile onnx_div_params_t *) params;
    for (int i = 0; i < LEN; i++) {
        offset = i * sizeof(float16);

        mmio_fp16(EXP_BASE + offset) = expected[i];
        mmio_fp16(SRC_A_BASE + offset) = vec_a[i];
        mmio_fp16(SRC_B_BASE + offset) = vec_b[i];
        mmio_fp16(RES_BASE + offset) = 0;
    }

    div_params->addr_a = SRC_A_BASE;
    div_params->addr_b = SRC_B_BASE;
    div_params->addr_res = RES_BASE;
    div_params->addr_exp = EXP_BASE;
    div_params->len = LEN;

    return 0;
}

static int run_spatz_task()
{
    int ret;

    eu_init();
    eu_enable_events(EU_SPATZ_DONE_MASK);



    spatz_init(SPATZ_BINARY_START);
    printf("[CV32] Random print just to slow down CV32 between Spatz init and run\n");  // TODOs: remove me
    spatz_run_task_with_params(ONNX_DIV_TASK, ONNX_DIV_PARAMS_BASE);

    eu_wait_spatz_wfe(EU_SPATZ_DONE_MASK);

    ret = spatz_get_exit_code();

    spatz_clk_dis();

    return ret;
}

static bool check_result(void *params)
{
    volatile onnx_div_params_t *div_params;
    div_params = (volatile onnx_div_params_t *) params;
    return vector_compare_fp16_bitwise(div_params->addr_res, div_params->addr_exp, div_params->len);
}

static bool run_test()
{
    int ret;
    bool check;
    volatile onnx_div_params_t *params;

    params = (volatile onnx_div_params_t *) ONNX_DIV_PARAMS_BASE;

    ret = init_data(params);
    if (ret != 0) {
        printf("[CV32] Params initialization failed with error: %d\n", ret);
        return ret;
    }

    ret = run_spatz_task();
    if (ret != 0) {
        printf("[CV32] Spatz task FAILED with error: %d", ret);
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

    printf("\n##################################### ONNX_DIV TEST #####################################\n\n");

    ret = run_test();

    printf("\n##########################################################################################\n\n");

    return ret;
}
