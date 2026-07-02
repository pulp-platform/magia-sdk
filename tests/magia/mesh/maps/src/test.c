#include <stdint.h>

#include "test.h"

#define WAIT_MODE WFE

int main(void)
{
    // ============================ Initializations ============================
    // Get tile ID
    uint32_t hartid = get_hartid();

    // Init iDMA
    idma_config_t idma_cfg      = {.hartid = hartid};
    idma_controller_t idma_ctrl = {
        .base = 0,
        .cfg  = &idma_cfg,
        .api  = &idma_api,
    };
    idma_init(&idma_ctrl);

    // Init RedMulE
    redmule_config_t redmule_cfg      = {.hartid = hartid};
    redmule_controller_t redmule_ctrl = {
        .base = 0,
        .cfg  = &redmule_cfg,
        .api  = &redmule_api,
    };
    redmule_init(&redmule_ctrl);

    // Init FractalSync
    fsync_config_t fsync_cfg      = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {
        .base = 0,
        .cfg  = &fsync_cfg,
        .api  = &fsync_api,
    };
    fsync_init(&fsync_ctrl);

    // Init EventUnit
    eu_config_t eu_cfg      = {.hartid = hartid};
    eu_controller_t eu_ctrl = {
        .base = 0,
        .cfg  = &eu_cfg,
        .api  = &eu_api,
    };

    eu_init(&eu_ctrl);
    eu_clear_events(0xFFFFFFFF);
    eu_fsync_init(&eu_ctrl, 0u);
    eu_idma_init(&eu_ctrl, 0u);
    eu_redmule_init(&eu_ctrl, 0u);

    /*
     * Startup barrier: ensure all tiles finish crt0.S BSS zeroing (which
     * touches global output buffers) and FIFO initialization before any
     * tile begins pushing into another tile's FIFO.
     */
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    // ============================ MAPS ============================
    // [PLACEHOLDER] Populate tensors
    maps_main_trace(hartid, "init-tensors");
    maps_generated_init_tensors(hartid);

    // [FIFO PLACEHOLDER] Clear mailbox flags
    maps_generated_clear_mailbox(hartid);

    // Sync barrier
    maps_main_trace(hartid, "barrier0");
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, MAPS_WAIT_MODE);

    // Prepare runtime context struct
    maps_generated_runtime_t runtime = {
        .redmule_ctrl = &redmule_ctrl,
        .eu_ctrl      = &eu_ctrl,
    };

    // [PLACEHOLDER] Create tile plan
    tile_plan_t plan;
    uint8_t plan_filled = maps_generated_fill_plan(&plan, hartid, &runtime);

    // Copy tile permanent data from L2 ("initializers" from MAPS, generally weights and biases)
    if (plan_filled) {
        maps_trace_event(&plan, 0u, 0u, "plan-active", 0u);
        init_tile(&plan, &idma_ctrl, &eu_ctrl);
    } else {
        maps_trace_event(&plan, 0u, 0u, "plan-inactive", 0u);
    }

    // Sync barrier
    maps_main_trace(hartid, "barrier1");
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, MAPS_WAIT_MODE);
    maps_main_trace(hartid, "barrier1-done");

    // Run tile plan
    if (plan_filled) {
        maps_trace_event(&plan, 0u, 0u, "run-begin", MAPS_NUM_TOKENS);
        run_tile_tokens(&plan, MAPS_NUM_TOKENS, &idma_ctrl, &eu_ctrl);
        maps_trace_event(&plan, 0u, 0u, "run-done", MAPS_NUM_TOKENS);
    }

    // Sync barrier
    maps_main_trace(hartid, "barrier2");
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, MAPS_WAIT_MODE);
    maps_main_trace(hartid, "barrier2-done");

    // [PLACEHOLDER] Check output
    if (hartid == 0u) {
        maps_main_trace(hartid, "check");
        uint32_t errors = maps_generated_check_output();
        printf("maps_handwritten errors: %d\n", errors);
        return errors;
    }

    return 0;
}
