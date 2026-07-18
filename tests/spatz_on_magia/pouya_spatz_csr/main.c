#include "tile.h"
#include "eventunit.h"

#include "compare_utils.h"
//#include "data.h"
#include "data_sample.h"
#include "onnx_spmv_mem_layout.h"
#include "onnx_spmv_params.h"
#include "spatz_csr_task_bin.h"

static int init_data(void *params)
{
    volatile onnx_spmv_params_t *spmv_params;

    spmv_params = (volatile onnx_spmv_params_t *)params;

    /* CSR row pointer */
    for (int i = 0; i <= DIM_M; i++) {
        mmio32(ROW_PTR_BASE + i * sizeof(uint32_t)) = row_ptr[i];
    }

    /* CSR column indices + values */
    for (int i = 0; i < NNZ; i++) {
        /* Pre-shift element index -> byte offset (fp16 = 2 bytes/elem) here
         * on the host, so the Spatz kernel never needs to touch vtype more
         * than once per row. */
        mmio32(COL_IDX_BASE + i * sizeof(uint32_t))  = col_idx[i] * sizeof(float16);
        mmio_fp16(VALUES_BASE + i * sizeof(float16)) = values[i];
    }

    /* Vector x */
    for (int i = 0; i < DIM_K; i++) {
        mmio_fp16(X_BASE + i * sizeof(float16)) = x[i];
    }

    /* Vector Y (computed) and G (golden) */
    for (int i = 0; i < DIM_M; i++) {
        mmio_fp16(G_BASE + i * sizeof(float16)) = G[i];
        mmio_fp16(Y_BASE + i * sizeof(float16)) = 0;
    }

    spmv_params->addr_row_ptr = ROW_PTR_BASE;
    spmv_params->addr_col_idx = COL_IDX_BASE;
    spmv_params->addr_values  = VALUES_BASE;
    spmv_params->addr_x       = X_BASE;
    spmv_params->addr_Y       = Y_BASE;
    spmv_params->addr_G       = G_BASE;
    spmv_params->M            = DIM_M;
    spmv_params->K            = DIM_K;
    spmv_params->nnz          = NNZ;

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
    spatz_run_task_with_params(ONNX_SPMV_TASK, ONNX_SPMV_PARAMS_BASE);

    eu_spatz_wait(&eu_ctrl, WFE);

    ret = spatz_get_exit_code();

    spatz_clk_dis();

    return ret;
}

static bool check_result(void *params)
{
    volatile onnx_spmv_params_t *spmv_params;
    spmv_params = (volatile onnx_spmv_params_t *)params;
    /* Treat the M-length output vector as an M x 1 matrix for comparison */
    return vector_compare_fp16_bitwise(
        spmv_params->addr_Y, spmv_params->addr_G, spmv_params->M);
}

static bool run_test()
{
    int ret;
    bool check;
    volatile onnx_spmv_params_t *params;

    params = (volatile onnx_spmv_params_t *)ONNX_SPMV_PARAMS_BASE;

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

    printf("\n################################### ONNX_SPMV TEST "
           "###################################\n\n");

    ret = run_test();

    printf("\n#####################################################################################"
           "#####\n\n");

    return ret;
}