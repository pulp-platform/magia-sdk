#include "tile.h"
#include "eventunit.h"

#include "data.h"
#include "vmac_fs_mem_layout.h"
#include "vmac_fs_params.h"
#include "vmac_fs_task_bin.h"

#define HID get_hartid()

/* Each hart runs entirely on its own: no cross-hart / cross-tile fsync
 * barrier, since the 5 tests are independent of what any other core does. */

static int init_data(void *params, int test_idx)
{
    volatile vmac_fs_params_t *vmac_params;

    vmac_params = (volatile vmac_fs_params_t *)params;

    for (int i = 0; i < VEC_LEN; i++) {
        mmio_fp32(CHUNK_A_BASE + (i * sizeof(float))) = A[test_idx][i];
        mmio_fp32(CHUNK_B_BASE + (i * sizeof(float))) = B[test_idx][i];
    }

    vmac_params->chunk_A = CHUNK_A_BASE;
    vmac_params->chunk_B = CHUNK_B_BASE;
    vmac_params->chunk_C = CHUNK_C_BASE;
    vmac_params->len     = VEC_LEN;

    return 0;
}

static int run_spatz_task(eu_controller_t *eu_ctrl, int init)
{
    int ret;

    // Only init the SPATZ core once; later calls reuse it so it doesn't
    // get stuck (same trick as the FFT example).
    if (!init)
        spatz_init(SPATZ_BINARY_START);

    spatz_run_task_with_params(VMAC_FS_TASK, VMAC_PARAMS_BASE);

    eu_spatz_wait(eu_ctrl, WFE);

    ret = spatz_get_exit_code();

    return ret;
}

static int run_test(void)
{
    int ret;
    int mismatches = 0;
    float result;
    volatile vmac_fs_params_t *params;

    params = (volatile vmac_fs_params_t *)VMAC_PARAMS_BASE;

    eu_config_t eu_cfg;
    eu_controller_t eu_ctrl;

    eu_cfg.hartid = get_hartid();
    eu_ctrl.base = NULL, eu_ctrl.cfg = &eu_cfg, eu_ctrl.api = &eu_api,

    eu_init(&eu_ctrl);
    eu_spatz_init(&eu_ctrl, 0);

    for (int t = 0; t < NUM_TESTS; t++) {
        ret = init_data((void *)params, t);
        if (ret != 0) {
            printf("[CV32 (%d)] Params init failed at test %d: %d\n", HID, t, ret);
            return -1;
        }

        ret = run_spatz_task(&eu_ctrl, t);
        if (ret != 0) {
            printf("[CV32 (%d)] Spatz task FAILED at test %d: %d\n", HID, t, ret);
            return -1;
        }

        result = mmio_fp32(params->chunk_C);

        if (result != GOLDEN[t]) {
            mismatches++;

        }
    }

    printf("[CV32 (%d)] Mismatches: %d / %d\n", HID, mismatches, NUM_TESTS);

    return mismatches;
}

int main(void)
{
    int ret;

    printf("\n############### VMAC_FS TEST on hart %d ###############\n\n", HID);

    ret = run_test();

    printf("\n#########################################################\n\n");

    return ret;
}