#include <stdint.h>

#include "eventunit.h"
#include "fsync.h"
#include "tile.h"

#include "kernels_compare_utils.h"

#include "network.h"
#include "data.h"

// TODO: use performace_utils.h API when rebased
static inline uint32_t read_cycles(void)
{
    uint32_t value;
    asm volatile("csrr %0, 0xB00" : "=r"(value));
    return value;
}

int init_fsync(fsync_controller_t *fsync_ctrl)
{
    fsync_config_t fsync_cfg;

    fsync_cfg.hartid = get_hartid();
    fsync_ctrl->base = NULL;
    fsync_ctrl->cfg = &fsync_cfg;
    fsync_ctrl->api = &fsync_api;

    fsync_init(fsync_ctrl);

    return 0;
}

int init_event_unit(eu_controller_t *eu_ctrl)
{
    eu_config_t eu_cfg;

    eu_cfg.hartid = get_hartid();
    eu_ctrl->base = NULL;
    eu_ctrl->cfg = &eu_cfg;
    eu_ctrl->api = &eu_api;

    eu_init(eu_ctrl);
    eu_spatz_init(eu_ctrl, 0);
    eu_fsync_init(eu_ctrl, 0);

    return 0;
}

int init_spatz()
{
    spatz_init(SPATZ_BINARY_START);

    return 0;
}

int deinit_spatz()
{
    spatz_clk_dis();

    return 0;
}

void sync(fsync_controller_t *fsynct_ctrl, eu_controller_t *eu_ctrl)
{
    fsync_sync_global(fsynct_ctrl);
    eu_fsync_wait(eu_ctrl, WFE);
}

void input_copy()
{
    for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
        memcpy(DeeployNetwork_inputs[buf], inputs[buf], DeeployNetwork_inputs_bytes[buf]);
    }
}

int check_result()
{
    int n_mismatch = 0;

    for (uint32_t i = 0; i < OUTPUTS_NUM; i++)
        n_mismatch += compare_fp16_bitwise((const float16 *)DeeployNetwork_outputs[i], (const float16 *)outputs[i], outputs_size[i]);

    return n_mismatch;
}

int main(void)
{
    fsync_controller_t fsync_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t cycle_start;
    uint32_t cycle_stop;
    int hid;
    int ret;

    hid = get_hartid();

    ret = init_fsync(&fsync_ctrl);
    if (ret) {
        printf("[CV32 (%d)] Fsync initialization failed with errno: %d\n", HID, ret);
        return ret;
    }

    ret = init_event_unit(&eu_ctrl);
    if (ret) {
        printf("[CV32 (%d)] Event Unit initialization failed with errno: %d\n", HID, ret);
        return ret;
    }

    ret = init_spatz();
    if (ret) {
        printf("[CV32 (%d)] Spatz initialization failed with errno: %d\n", HID, ret);
        return ret;
    }

    sync(&fsync_ctrl, &eu_ctrl);

    if (hid == 0)
        InitNetwork();

    sync(&fsync_ctrl, &eu_ctrl);

    if (hid == 0)
        input_copy();

    sync(&fsync_ctrl, &eu_ctrl);

    cycle_start = read_cycles();
    RunNetwork();
    cycle_stop = read_cycles();

    printf("[CV32 (%d)] Run completed in %d cycles\n", hid, cycle_stop - cycle_start);

    sync(&fsync_ctrl, &eu_ctrl);

    ret = deinit_spatz();
    if (ret) {
        printf("[CV32 (%d)] Spatz deinitialization failed with errno: %d\n", HID, ret);
        return ret;
    }

    if (hid == 0) {
        ret = check_result();
        printf("[CV32] Test completed with %d mismatches\n", ret);
    }

    sync(&fsync_ctrl, &eu_ctrl);

    return ret;
}
