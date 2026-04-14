#include "tile.h"
#include "eventunit.h"

#include "compare_utils.h"
#include "data.h"
#include "onnx_hardsigmoid_mem_layout.h"
#include "onnx_hardsigmoid_params.h"
#include "onnx_hardsigmoid_task_bin.h"

#define HID get_hartid()

static int init_data(void *params)
{
    volatile onnx_hardsigmoid_params_t *hardsigmoid_params;
    uint32_t start;
    uint32_t chunk;
    uint32_t left;
    uint32_t len;
    uint32_t end;

    hardsigmoid_params = (volatile onnx_hardsigmoid_params_t *) params;

    chunk = TENSOR_LEN / NUM_HARTS;
    left = TENSOR_LEN % NUM_HARTS;
    start = HID * chunk + (HID < left ? HID : left);
    end = start + chunk + (HID < left ? 1 : 0);
    len = end - start;

    for (int i = 0; i < len; i++) {
        int global_idx;
        uint32_t offset;

        global_idx = start + i;
        offset =  i * sizeof(float16);

        mmio_fp16(CHUNK_X_BASE + offset) = X[global_idx];
        mmio_fp16(CHUNK_G_BASE + offset) = G[global_idx];
        mmio_fp16(CHUNK_Y_BASE + offset) = 0;
    }

    mmio_fp16(ALPHA_BASE) = alpha;
    mmio_fp16(BETA_BASE) = beta;

    hardsigmoid_params->chunk_X = CHUNK_X_BASE;
    hardsigmoid_params->chunk_Y = CHUNK_Y_BASE;
    hardsigmoid_params->chunk_G = CHUNK_G_BASE;
    hardsigmoid_params->alpha = ALPHA_BASE;
    hardsigmoid_params->beta = BETA_BASE;
    hardsigmoid_params->start = start;
    hardsigmoid_params->len = len;
    hardsigmoid_params->end = end;

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
    spatz_run_task_with_params(ONNX_HARDSIGMOID_TASK, ONNX_HARDSIGMOID_PARAMS_BASE);

    eu_spatz_wait(&eu_ctrl, WFE);

    ret = spatz_get_exit_code();

    spatz_clk_dis();

    return ret;
}

static bool check_result(void *params)
{
    volatile onnx_hardsigmoid_params_t *hardsigmoid_params;
    hardsigmoid_params = (volatile onnx_hardsigmoid_params_t *) params;
    return chunk_compare_fp16_bitwise(hardsigmoid_params->chunk_Y, hardsigmoid_params->chunk_G, hardsigmoid_params->start, hardsigmoid_params->len);
}

static bool run_test()
{
    int ret;
    bool check;
    volatile onnx_hardsigmoid_params_t *params;

    params = (volatile onnx_hardsigmoid_params_t *) ONNX_HARDSIGMOID_PARAMS_BASE;

    ret = init_data(params);
    if (ret != 0) {
        printf("[CV32 (%d)] Params initialization failed with error: %d\n", HID, ret);
        return ret;
    }

    ret = run_spatz_task();
    if (ret != 0) {
        printf("[CV32 (%d)] Spatz task FAILED with error: %d", HID, ret);
        return ret;
    }

    check = check_result(params);
    if (check) {
        printf("[CV32 (%d)] Test SUCCESS\n", HID);
    } else {
        printf("[CV32 (%d)] Test FAILED\n", HID);
        ret = -1;
    }

    return ret;
}

int main(void)
{
    int ret;

    if (HID == 0) printf("\n############################### ONNX_HARDSIGMOID TEST on %d Tiles ################################\n\n", NUM_HARTS);

    ret = run_test();

    if (HID == 0) printf("\n##########################################################################################\n\n");

    return ret;
}
