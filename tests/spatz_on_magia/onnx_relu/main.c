#include "tile.h"
#include "eventunit.h"

#include "compare_utils.h"
#include "data.h"
#include "onnx_relu_mem_layout.h"
#include "onnx_relu_params.h"
#include "onnx_relu_task_bin.h"

static int init_data(void *params)
{
    uint32_t offset;
    volatile onnx_relu_params_t *relu_params;

    relu_params = (volatile onnx_relu_params_t *) params;
    for (int i = 0; i < LEN; i++) {
        offset = i * sizeof(float16);

        mmio_fp16(EXP_BASE + offset) = expected_vec[i];
        mmio_fp16(INPUT_BASE + offset) = input_vec[i];
        mmio_fp16(RES_BASE + offset) = 0;
    }

    relu_params->addr_src = INPUT_BASE;
    relu_params->addr_res = RES_BASE;
    relu_params->addr_exp = EXP_BASE;
    relu_params->len = LEN;

    return 0;
}

static int run_spatz_task()
{
    int ret;
    eu_config_t eu_cfg;
    eu_controller_t eu_ctrl;

    eu_cfg.hartid = get_hartid();
    eu_ctrl.base = NULL;
    eu_ctrl.cfg = &eu_cfg;
    eu_ctrl.api = &eu_api;

    eu_init(&eu_ctrl);
    eu_spatz_init(&eu_ctrl, 0);

    spatz_init(SPATZ_BINARY_START);
    spatz_run_task_with_params(ONNX_RELU_TASK, ONNX_RELU_PARAMS_BASE);

    eu_spatz_wait(&eu_ctrl, WFE);

    ret = spatz_get_exit_code();

    spatz_clk_dis();

    return ret;
}

static bool check_result(void *params)
{
    volatile onnx_relu_params_t *relu_params;
    relu_params = (volatile onnx_relu_params_t *) params;
    return vector_compare_fp16_bitwise(relu_params->addr_res, relu_params->addr_exp, relu_params->len);
}

static bool run_test()
{
    int ret;
    bool check;
    volatile onnx_relu_params_t *params;

    params = (volatile onnx_relu_params_t *) ONNX_RELU_PARAMS_BASE;

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

    printf("\n################################### ONNX_RELU TEST ###################################\n\n");

    ret = run_test();

    printf("\n##########################################################################################\n\n");

    return ret;
}
