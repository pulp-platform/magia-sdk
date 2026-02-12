#include <stdint.h>

#include "compare_utils.h"
#include "data.h"
#include "event_unit_utils.h"
#include "onnx_add_mem_layout.h"
#include "onnx_add_params.h"
#include "onnx_add_task_bin.h"

#include "magia_tile_utils.h"
#include "magia_spatz_utils.h"

volatile onnx_add_params_t *params;

static void init_data()
{
    uint32_t offset;

    for (int i = 0; i < LEN; i++) {
        offset = i * sizeof(float16);

        mmio_fp16(EXP_BASE + offset) = expected[i];
        mmio_fp16(SRC_A_BASE + offset) = vec_a[i];
        mmio_fp16(SRC_B_BASE + offset) = vec_b[i];
        mmio_fp16(RES_BASE + offset) = 0;
    }

    params = (volatile onnx_add_params_t *) ONNX_ADD_PARAMS_BASE;
    params->addr_a = SRC_A_BASE;
    params->addr_b = SRC_B_BASE;
    params->addr_res = RES_BASE;
    params->addr_exp = EXP_BASE;
    params->len = LEN;
}

static int run_spatz_task()
{
    int ret;

    eu_init();
    eu_enable_events(EU_SPATZ_DONE_MASK);

    spatz_init(SPATZ_BINARY_START);
    spatz_run_task_with_params(ONNX_ADD_TASK, ONNX_ADD_PARAMS_BASE);

    eu_wait_spatz_wfe(EU_SPATZ_DONE_MASK);

    ret = spatz_get_exit_code();

    spatz_clk_dis();

    return ret;
}

static bool check_result()
{
    return vector_compare_fp16(params->addr_res, params->addr_exp, params->addr_exp);
}

static bool run_test()
{
    int ret;
    bool check;

    init_data();

    ret = run_spatz_task();
    if (ret != 0) {
        printf("[CV32] Spatz task FAILED with error: %d", ret);
        return ret;
    }

    check = check_result();
    if (check) {
        printf("[CV32] Test SUCCESS\n");
    } else {
        printf("[CV32 Test FAILED\n]");
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
