#include "tile.h"
#include "eventunit.h"

#include "compare_utils.h"
#include "data.h"
#include "onnx_gemv_mem_layout.h"
#include "onnx_gemv_params.h"
#include "spatz_gemv2_task_bin.h"

static int init_data(void *params)
{
    volatile onnx_gemv_params_t *gemv_params;

    gemv_params = (volatile onnx_gemv_params_t *)params;

    /* Matrix A */
    for (int i = 0; i < DIM_M * DIM_K; i++) {
        mmio_fp16(A_BASE + i * sizeof(float16)) = A[i];
    }

    /* Vector x */
    for (int i = 0; i < DIM_K; i++) {
        mmio_fp16(X_BASE + i * sizeof(float16)) = x[i];
    }

    /* Vector C, Y (computed) and G (golden) */
    for (int i = 0; i < DIM_M; i++) {
        mmio_fp16(C_BASE + i * sizeof(float16)) = C[i];
        mmio_fp16(G_BASE + i * sizeof(float16)) = G[i];
        mmio_fp16(Y_BASE + i * sizeof(float16)) = 0;
    }

    mmio_fp16(ALPHA_BASE) = ALPHA;
    mmio_fp16(BETA_BASE)  = BETA;

    gemv_params->addr_alpha = ALPHA_BASE;
    gemv_params->addr_beta  = BETA_BASE;
    gemv_params->transA     = TRANS_A;
    gemv_params->addr_A     = A_BASE;
    gemv_params->addr_x     = X_BASE;
    gemv_params->addr_C     = C_BASE;
    gemv_params->addr_Y     = Y_BASE;
    gemv_params->addr_G     = G_BASE;
    gemv_params->M          = DIM_M;
    gemv_params->K          = DIM_K;

    return 0;
}

static int run_spatz_task()
{
    int ret;
    eu_config_t eu_cfg;
    eu_controller_t eu_ctrl;

    eu_cfg.hartid = get_hartid();
    eu_ctrl.base  = NULL;
    eu_ctrl.cfg   = &eu_cfg;
    eu_ctrl.api   = &eu_api;

    eu_init(&eu_ctrl);
    eu_spatz_init(&eu_ctrl, 0);

    spatz_init(SPATZ_BINARY_START);
    spatz_run_task_with_params(ONNX_GEMV_TASK, ONNX_GEMV_PARAMS_BASE);

    eu_spatz_wait(&eu_ctrl, WFE);

    ret = spatz_get_exit_code();

    spatz_clk_dis();

    return ret;
}

static bool check_result(void *params)
{
    volatile onnx_gemv_params_t *gemv_params;
    gemv_params = (volatile onnx_gemv_params_t *)params;
    /* Treat the M-length output vector as an M x 1 matrix for comparison */
    return matrix_compare_fp16_bitwise(
        gemv_params->addr_Y, gemv_params->addr_G, gemv_params->M, 1);
}

static bool run_test()
{
    int ret;
    bool check;
    volatile onnx_gemv_params_t *params;

    params = (volatile onnx_gemv_params_t *)ONNX_GEMV_PARAMS_BASE;

    ret = init_data((void *)params);
    if (ret != 0) {
        printf("[CV32] Params initialization failed with error: %d\n", ret);
        return ret;
    }

    ret = run_spatz_task();
    if (ret != 0) {
        printf("[CV32] Spatz task FAILED with error: %d", ret);
        return ret;
    }

    uint32_t hartid = get_hartid();

    if (hartid==0){

        check = check_result((void *)params);
        if (check) {
            printf("[CV32] Test SUCCESS\n");
        } else {
            printf("[CV32] Test FAILED\n");
            ret = -1;
        }
    }

    return ret;
}

int main(void)
{
    int ret;

    printf("\n################################### ONNX_GEMV TEST "
           "###################################\n\n");

    ret = run_test();

    printf("\n#####################################################################################"
           "#####\n\n");

    return ret;
}