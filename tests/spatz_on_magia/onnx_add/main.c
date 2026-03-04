#include "tile.h"
#include "eventunit.h"

#include "data.h"
#include "onnx_add_mem_layout.h"
#include "onnx_add_params.h"
#include "onnx_add_task_bin.h"

static int init_data(void *params)
{
    uint32_t offset;
    volatile onnx_add_params_t *add_params;

    add_params = (volatile onnx_add_params_t *) params;
    for (int i = 0; i < LEN; i++) {
        offset = i * sizeof(float16);

        mmio_fp16(EXP_BASE + offset) = expected[i];
        mmio_fp16(SRC_A_BASE + offset) = vec_a[i];
        mmio_fp16(SRC_B_BASE + offset) = vec_b[i];
        mmio_fp16(RES_BASE + offset) = 0;
    }


    add_params->addr_a = SRC_A_BASE;
    add_params->addr_b = SRC_B_BASE;
    add_params->addr_res = RES_BASE;
    add_params->addr_exp = EXP_BASE;
    add_params->len = LEN;

    return 0;
}

static int run_spatz_task()
{
    int ret;
    eu_config_t eu_cfg;
    eu_controller_t eu_ctrl;

    eu_cfg.hartid = get_hartid();
    eu_ctrl.base = NULL,
    eu_ctrl.cfg = &eu_cfg,
    eu_ctrl.api = &eu_api,

    eu_init(&eu_ctrl);
    eu_spatz_init(&eu_ctrl, 0);

    spatz_init(SPATZ_BINARY_START);
    spatz_run_task_with_params(ONNX_ADD_TASK, ONNX_ADD_PARAMS_BASE);

    eu_spatz_wait(&eu_ctrl, WFE);

    ret = spatz_get_exit_code();

    spatz_clk_dis();

    return ret;
}

static bool check_result(void *params)
{
    volatile onnx_add_params_t *add_params;
    add_params = (volatile onnx_add_params_t *) params;
    return vector_compare_fp16_bitwise(add_params->addr_res, add_params->addr_exp, add_params->len);
}

static bool run_test()
{
    int ret;
    bool check;
    volatile onnx_add_params_t *params;

    params = (volatile onnx_add_params_t *) ONNX_ADD_PARAMS_BASE;

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

    printf("\n##################################### ONNX_ADD TEST #####################################\n\n");

    ret = run_test();

    printf("\n##########################################################################################\n\n");

    return ret;
}
