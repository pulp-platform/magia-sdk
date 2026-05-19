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

static size_t bit_reverse(size_t x, size_t bits)
{
    size_t y = 0;

    for (size_t i = 0; i < bits; i++) {
        y = (y << 1) | (x & 1u);
        x >>= 1;
    }

    return y;
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

    start = HID * chunk;
    end = start + chunk;
    len = chunk;

    for(int i = 0; i < (VEC_LEN / 2); i++){
        mmio_fp16(CHUNK_AR_BASE + (i * sizeof(float16))) = IR[bit_reverse(i * 2, LOG2_LEN)];
        mmio_fp16(CHUNK_BR_BASE + (i * sizeof(float16))) = IR[bit_reverse(((i * 2) + 1), LOG2_LEN)];
        mmio_fp16(CHUNK_AI_BASE + (i * sizeof(float16))) = II[bit_reverse(i * 2, LOG2_LEN)];
        mmio_fp16(CHUNK_BI_BASE + (i * sizeof(float16))) = II[bit_reverse(((i * 2) + 1), LOG2_LEN)];
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
    fft_fs_params->permute = 0;

    return 0;
}

static int reinit_data(void *params, int stage)
{
    int i, j, k;
    int prev_stage = (1 << (stage - 1));
    // Number of different butterfly groups and number of elements
    int n_groups = (VEC_LEN >> (stage + 1));
    int group_len = (1 << stage);
    for(j = 0; j < n_groups; j++){
        for(i = 0; i < group_len; i++){
            if(!((i / prev_stage) % 2)){
                mmio_fp16(CHUNK_AR_BASE + (((j * group_len) + i) * sizeof(float16))) = mmio_fp16(CHUNK_CR_BASE + (2 * j * prev_stage + (i % prev_stage)) * sizeof(float16));
                mmio_fp16(CHUNK_AI_BASE + (((j * group_len) + i) * sizeof(float16))) = mmio_fp16(CHUNK_CI_BASE + (2 * j * prev_stage + (i % prev_stage)) * sizeof(float16));
                mmio_fp16(CHUNK_BR_BASE + (((j * group_len) + i) * sizeof(float16))) = mmio_fp16(CHUNK_CR_BASE + (((2 * j) + 1) * prev_stage + (i % prev_stage)) * sizeof(float16));
                mmio_fp16(CHUNK_BI_BASE + (((j * group_len) + i) * sizeof(float16))) = mmio_fp16(CHUNK_CI_BASE + (((2 * j) + 1) * prev_stage + (i % prev_stage)) * sizeof(float16));
            }
            else{
                mmio_fp16(CHUNK_AR_BASE + (((j * group_len) + i) * sizeof(float16))) = mmio_fp16(CHUNK_DR_BASE + (2 * j * prev_stage + (i % prev_stage)) * sizeof(float16));
                mmio_fp16(CHUNK_AI_BASE + (((j * group_len) + i) * sizeof(float16))) = mmio_fp16(CHUNK_DI_BASE + (2 * j * prev_stage + (i % prev_stage)) * sizeof(float16));
                mmio_fp16(CHUNK_BR_BASE + (((j * group_len) + i) * sizeof(float16))) = mmio_fp16(CHUNK_DR_BASE + (((2 * j) + 1) * prev_stage + (i % prev_stage)) * sizeof(float16));
                mmio_fp16(CHUNK_BI_BASE + (((j * group_len) + i) * sizeof(float16))) = mmio_fp16(CHUNK_DI_BASE + (((2 * j) + 1) * prev_stage + (i % prev_stage)) * sizeof(float16));
            }
        }
    }

    volatile fft_fs_params_t *fft_fs_params;
    fft_fs_params = (volatile fft_fs_params_t *) params;

    fft_fs_params->chunk_WR = fft_fs_params->chunk_WR + (VEC_LEN);
    fft_fs_params->chunk_WI = fft_fs_params->chunk_WI + (VEC_LEN);

    return 0;
}



static int run_spatz_task(eu_controller_t* eu_ctrl, int init)
{
    int ret;

    if(!init)
        spatz_init(SPATZ_BINARY_START);
    // else
    //     spatz_clk_en();
    spatz_run_task_with_params(FFT_FS_TASK, FFT_PARAMS_BASE);

    eu_spatz_wait(eu_ctrl, WFE);

    ret = spatz_get_exit_code();

    //spatz_clk_dis();

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

    eu_config_t eu_cfg;
    eu_controller_t eu_ctrl;

    eu_cfg.hartid = get_hartid();
    eu_ctrl.base = NULL,
    eu_ctrl.cfg = &eu_cfg,
    eu_ctrl.api = &eu_api,

    eu_init(&eu_ctrl);
    eu_spatz_init(&eu_ctrl, 0);

    ret = init_data((void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] Params initialization failed with error: %d\n", HID, ret);
        return ret;
    }

    for(int i = 0; i < LOG2_LEN; i++){
        printf("Complex mul number %d\n", i);
        ret = run_spatz_task(&eu_ctrl, i);
        if (ret != 0) {
            printf("[CV32 (%d)] Spatz task FAILED with error: %d", HID, ret);
            return ret;
        }

        if(i != (LOG2_LEN - 1)){
            printf("Swapping at i=%d\n", i);
            ret = reinit_data((void *)params, (i + 1));
            if (ret != 0) {
                printf("[CV32 (%d)] Params swapping failed with error: %d\n", HID, ret);
                return ret;
            }
        }
    }

    printf("CR VECTOR:\n");
    print_vector_raw((const float16*) params->chunk_CR, (size_t) params->len);
    printf("CI VECTOR:\n");
    print_vector_raw((const float16*) params->chunk_CI, (size_t) params->len);
    printf("DR VECTOR:\n");
    print_vector_raw((const float16*) params->chunk_DR, (size_t) params->len);
    printf("DI VECTOR:\n");
    print_vector_raw((const float16*) params->chunk_DI, (size_t) params->len);
    

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
