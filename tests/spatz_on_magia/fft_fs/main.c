#include "tile.h"
#include "eventunit.h"
#include "fsync.h"

#include "compare_utils.h"
#include "data.h"
#include "fft_fs_mem_layout.h"
#include "fft_fs_params.h"
#include "fft_fs_task_bin.h"

#define HID get_hartid()

static inline uint16_t get_raw(const float16 val)
{
    uint16_t raw;
    memcpy(&raw, &val, sizeof(raw));
    return raw;
}
 
static inline void print_vector_raw(const float16 *vec, size_t len)
{
    for (size_t i = 0; i < len; i++) {
        printf("%d) %x\n", i, get_raw(vec[i]));
    }
}

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

    printf("IR vector:\n");
    print_vector_raw(IR, VEC_LEN);
    printf("II vector:\n");
    print_vector_raw(II, VEC_LEN);
    printf("WR vector:\n");
    print_vector_raw(WR, TW_LEN);
    printf("WI vector:\n");
    print_vector_raw(WI, TW_LEN);

    printf("CHUNK_AR_BASE: %d\n", CHUNK_AR_BASE);
    printf("CHUNK_AR_SIZE: %d\n", CHUNK_AR_SIZE);
    printf("CHUNK_AI_BASE: %d\n", CHUNK_AI_BASE);
    printf("CHUNK_AI_SIZE: %d\n", CHUNK_AI_SIZE);

    for(int i = 0; i < (VEC_LEN / 2); i++){
        mmio_fp16(CHUNK_AR_BASE + (i * sizeof(float16))) = IR[i * 2];
        mmio_fp16(CHUNK_BR_BASE + (i * sizeof(float16))) = IR[((i * 2) + 1)];
        mmio_fp16(CHUNK_AI_BASE + (i * sizeof(float16))) = II[i * 2];
        mmio_fp16(CHUNK_BI_BASE + (i * sizeof(float16))) = II[((i * 2) + 1)];
    }

    for(int i = 0; i < TW_LEN; i++){
        mmio_fp16(CHUNK_WR_BASE + (i * sizeof(float16))) = WR[i];
        mmio_fp16(CHUNK_WI_BASE + (i * sizeof(float16))) = WI[i];
    }

    fft_fs_params->chunk_AR = CHUNK_AR_BASE;
    fft_fs_params->chunk_AI = CHUNK_AI_BASE;
    fft_fs_params->chunk_BR = CHUNK_BR_BASE;
    fft_fs_params->chunk_BI = CHUNK_BI_BASE;
    fft_fs_params->chunk_WR = CHUNK_WR_BASE;
    fft_fs_params->chunk_WI = CHUNK_WI_BASE;
    fft_fs_params->chunk_CR = CHUNK_CR_BASE;
    fft_fs_params->chunk_CI = CHUNK_CI_BASE;
    fft_fs_params->chunk_DR = CHUNK_DR_BASE;
    fft_fs_params->chunk_DI = CHUNK_DI_BASE;
    fft_fs_params->start = start;
    fft_fs_params->len = (VEC_LEN / 2);
    fft_fs_params->end = end;

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
    spatz_run_task_with_params(FFT_FS_TASK, FFT_PARAMS_BASE);

    eu_spatz_wait(&eu_ctrl, WFE);

    ret = spatz_get_exit_code();

    spatz_clk_dis();

    return ret;
}

// static bool check_result(void *params)
// {
//     volatile fft_fparams_t *fft_params;
//     fft_params = (volatile fft_params_t *) params;
//     return chunk_compare_fp16_bitwise(fft_params->chunk_CR, fft_params->chunk_GR, fft_params->chunk_CI, fft_params->chunk_GI, fft_params->start, fft_params->len);
// }

static bool run_test()
{
    int ret;
    bool check;
    volatile fft_fs_params_t *params;

    params = (volatile fft_fs_params_t *) FFT_PARAMS_BASE;

    ret = init_data(params);
    if (ret != 0) {
        printf("[CV32 (%d)] Params initialization failed with error: %d\n", HID, ret);
        return ret;
    }

    // printf("AR VECTOR:\n");
    // print_vector_raw((const float16*) params->chunk_AR, (size_t) params->len);
    // printf("AI VECTOR:\n");
    // print_vector_raw((const float16*) params->chunk_AI, (size_t) params->len);
    // printf("BR VECTOR:\n");
    // print_vector_raw((const float16*) params->chunk_BR, (size_t) params->len);
    // printf("BI VECTOR:\n");
    // print_vector_raw((const float16*) params->chunk_BI, (size_t) params->len);

    // printf("WR VECTOR:\n");
    // print_vector_raw((const float16*) params->chunk_WR, (size_t) params->len);
    // printf("WI VECTOR:\n");
    // print_vector_raw((const float16*) params->chunk_WI, (size_t) params->len);

    // printf("CR VECTOR:\n");
    // print_vector_raw((const float16*) params->chunk_CR, (size_t) params->len);
    // printf("CI VECTOR:\n");
    // print_vector_raw((const float16*) params->chunk_CI, (size_t) params->len);
    // printf("DR VECTOR:\n");
    // print_vector_raw((const float16*) params->chunk_DR, (size_t) params->len);
    // printf("DI VECTOR:\n");
    // print_vector_raw((const float16*) params->chunk_DI, (size_t) params->len);

    ret = run_spatz_task();
    if (ret != 0) {
        printf("[CV32 (%d)] Spatz task FAILED with error: %d", HID, ret);
        return ret;
    }

    // check = check_result(params);
    // if (check) {
    //     printf("[CV32 (%d)] Test SUCCESS\n", HID);
    // } else {
    //     printf("[CV32 (%d)] Test FAILED\n", HID);
    //     ret = -1;
    // }

    return ret;
}

int main(void)
{

    fsync_config_t fsync_cfg = {.hartid = HID};
    fsync_controller_t fsync_ctrl = {
        .base = NULL,
        .cfg = &fsync_cfg,
        .api = &fsync_api,
    };

    fsync_init(&fsync_ctrl);

    eu_config_t eu_cfg = {.hartid = HID};
    eu_controller_t eu_ctrl = {
        .base = NULL,
        .cfg = &eu_cfg,
        .api = &eu_api,
    };
    eu_init(&eu_ctrl);
    eu_fsync_init(&eu_ctrl, 0);

    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WFE);

    int ret;

    if (HID == 0) {printf("\n############################### FFT_FS TEST on %d Tiles ################################\n\n", NUM_HARTS);

    ret = run_test();

    if (HID == 0) printf("\n##########################################################################################\n\n");
    }
    return ret;
}
