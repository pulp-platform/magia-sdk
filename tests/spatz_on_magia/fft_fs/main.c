#include "tile.h"
#include "eventunit.h"

#include "compare_utils.h"
#include "data.h"
#include "fft_fs_mem_layout.h"
#include "fft_fs_params.h"
#include "fft_fs_task_bin.h"

#define HID get_hartid()

static int init_data(void *params)
{
    volatile fft_fs_params_t *fft_fs_params;
    uint32_t start;
    uint32_t chunk;
    uint32_t left;
    uint32_t len;
    uint32_t end;

    fft_fs_params = (volatile fft_fs_params_t *) params;

    // chunk = TENSOR_LEN / NUM_HARTS;
    // left = TENSOR_LEN % NUM_HARTS;
    // start = HID * chunk + (HID < left ? HID : left);
    // end = start + chunk + (HID < left ? 1 : 0);
    // len = end - start;

    //chunk = TENSOR_LEN;
    start = HID * chunk;
    end = start + chunk;
    len = chunk;

    // for (int i = 0; i < len; i++) {
    //     int global_idx;
    //     uint32_t offset;

    //     global_idx = start + i;
    //     offset =  i * sizeof(float16);

    //     mmio_fp16(CHUNK_AR_BASE + offset)    =  AR[global_idx];
    //     mmio_fp16(CHUNK_BR_BASE + offset)    =   B[gloabl_idx];
    //     mmio_fp16(CHUNK_AI_BASE + offset)    =   A[global_idx];
    //     mmio_fp16(CHUNK_BI_BASE + offset)    =   B[gloabl_idx];
    //     mmio_fp16(CHUNK_WR_BASE + offset)   =   WR[global_idx];
    //     mmio_fp16(CHUNK_GR_BASE + offset)   =   GR[global_idx];
    //     mmio_fp16(CHUNK_WI_BASE + offset)   =   WI[global_idx];
    //     mmio_fp16(CHUNK_GI_BASE + offset)   =   GI[global_idx];
    //     mmio_fp16(CHUNK_CR_BASE + offset)   =   0;
    //     mmio_fp16(CHUNK_CI_BASE + offset)   =   0;
    // }

    for(int i = 0; i < (VEC_LEN / 2); i++){
        mmio_fp16(CHUNK_AR_BASE + (i * sizeof(float16))) = IR[i * 2];
        mmio_fp16(CHUNK_BR_BASE + (i * sizeof(float16))) = IR[i * 2 + 1];
        mmio_fp16(CHUNK_AI_BASE + (i * sizeof(float16))) = II[i * 2];
        mmio_fp16(CHUNK_BI_BASE + (i * sizeof(float16))) = II[i * 2 + 1];
    }

    for(int i = 0; i < TW_LEN; i++){
        mmio_fp16(CHUNK_WR_BASE + (i * sizeof(float16))) = WR[i];
        mmio_fp16(CHUNK_WI_BASE + (i * sizeof(float16))) = WI[i];
    }

    fft_params->chunk_AR = CHUNK_AR_BASE;
    fft_params->chunk_AI = CHUNK_AI_BASE;
    fft_params->chunk_BR = CHUNK_BR_BASE;
    fft_params->chunk_BI = CHUNK_BI_BASE;
    fft_params->chunk_WR = CHUNK_WR_BASE;
    fft_params->chunk_WI = CHUNK_WI_BASE;
    fft_params->chunk_CR = CHUNK_CR_BASE;
    fft_params->chunk_CI = CHUNK_CI_BASE;
    fft_params->chunk_DR = CHUNK_DR_BASE;
    fft_params->chunk_DI = CHUNK_DI_BASE;
    fft_params->start = start;
    fft_params->len = len;
    fft_params->end = end;

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
    spatz_run_task_with_params(FFT_TASK, FFT_PARAMS_BASE);

    eu_spatz_wait(&eu_ctrl, WFE);

    ret = spatz_get_exit_code();

    spatz_clk_dis();

    return ret;
}

static bool check_result(void *params)
{
    volatile fft_params_t *fft_params;
    fft_params = (volatile fft_params_t *) params;
    return chunk_compare_fp16_bitwise(fft_params->chunk_CR, fft_params->chunk_GR, fft_params->chunk_CI, fft_params->chunk_GI, fft_params->start, fft_params->len);
}

static bool run_test()
{
    int ret;
    bool check;
    volatile fft_params_t *params;

    params = (volatile fft_params_t *) FFT_PARAMS_BASE;

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

    if (HID == 0) printf("\n############################### FFT_FS TEST on %d Tiles ################################\n\n", NUM_HARTS);

    ret = run_test();

    if (HID == 0) printf("\n##########################################################################################\n\n");

    return ret;
}
