#include "tile.h"
#include "eventunit.h"

#include "compare_utils.h"
#include "data.h"
#include "onnx_gelu_mem_layout.h"
#include "onnx_gelu_params.h"
#include "onnx_gelu_task_bin.h"

static int init_data(void *params)
{
    uint32_t offset;
    volatile onnx_gelu_params_t *gelu_params;

    gelu_params = (volatile onnx_gelu_params_t *) params;
    for (int i = 0; i < LEN; i++) {
        offset = i * sizeof(float16);

        mmio_fp16(EXP_BASE + offset) = expected_vec[i];
        mmio_fp16(INPUT_BASE + offset) = input_vec[i];
        mmio_fp16(RES_BASE + offset) = 0;
    }

    gelu_params->addr_src = INPUT_BASE;
    gelu_params->addr_res = RES_BASE;
    gelu_params->addr_exp = EXP_BASE;
    gelu_params->len = LEN;

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
    spatz_run_task_with_params(ONNX_GELU_TASK, ONNX_GELU_PARAMS_BASE);

    eu_spatz_wait(&eu_ctrl, WFE);

    ret = spatz_get_exit_code();

    spatz_clk_dis();

    return ret;
}

static bool check_result(void *params)
{
    volatile onnx_gelu_params_t *gelu_params;
    gelu_params = (volatile onnx_gelu_params_t *) params;
    return vector_compare_fp16_bitwise(gelu_params->addr_res, gelu_params->addr_exp, gelu_params->len);
}

static bool run_test()
{
    int ret;
    bool check;
    volatile onnx_gelu_params_t *params;

    params = (volatile onnx_gelu_params_t *) ONNX_GELU_PARAMS_BASE;

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

    printf("\n#################################### ONNX_GELU TEST #####################################\n\n");

    ret = run_test();

    printf("\n##########################################################################################\n\n");

    return ret;
}
