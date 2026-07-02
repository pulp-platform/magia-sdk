#include <stdint.h>

#include "test.h"

static inline void maps_main_trace(uint32_t hartid, const char *phase)
{
#if MAPS_ENABLE_TRACE
    if (hartid == 0u) {
        printf("maps main %s\n", phase);
    }
#else
    (void)hartid;
    (void)phase;
#endif
}

int main(void)
{
    uint32_t hartid = get_hartid();

    idma_config_t idma_cfg      = {.hartid = hartid};
    idma_controller_t idma_ctrl = {
        .base = 0,
        .cfg  = &idma_cfg,
        .api  = &idma_api,
    };

    redmule_config_t redmule_cfg      = {.hartid = hartid};
    redmule_controller_t redmule_ctrl = {
        .base = 0,
        .cfg  = &redmule_cfg,
        .api  = &redmule_api,
    };

    fsync_config_t fsync_cfg      = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {
        .base = 0,
        .cfg  = &fsync_cfg,
        .api  = &fsync_api,
    };

    eu_config_t eu_cfg      = {.hartid = hartid};
    eu_controller_t eu_ctrl = {
        .base = 0,
        .cfg  = &eu_cfg,
        .api  = &eu_api,
    };

    idma_init(&idma_ctrl);
    redmule_init(&redmule_ctrl);
    fsync_init(&fsync_ctrl);
    eu_init(&eu_ctrl);
    eu_idma_init(&eu_ctrl, 0u);
    eu_redmule_init(&eu_ctrl, 0u);
    eu_fsync_init(&eu_ctrl, 0u);

    maps_main_trace(hartid, "init-tensors");
    maps_generated_init_tensors(hartid);
    maps_generated_clear_mailbox(hartid);

    maps_main_trace(hartid, "barrier0");
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, MAPS_WAIT_MODE);
    maps_main_trace(hartid, "barrier0-done");

    maps_generated_runtime_t runtime = {
        .redmule_ctrl = &redmule_ctrl,
        .eu_ctrl      = &eu_ctrl,
    };

    tile_plan_t plan;
    if (maps_generated_fill_plan(&plan, hartid, &runtime)) {
        maps_trace_event(&plan, 0u, 0u, "plan-active", 0u);
        init_tile(&plan, &idma_ctrl, &eu_ctrl);
    }

    maps_main_trace(hartid, "barrier1");
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, MAPS_WAIT_MODE);
    maps_main_trace(hartid, "barrier1-done");

    if (maps_generated_fill_plan(&plan, hartid, &runtime)) {
        maps_trace_event(&plan, 0u, 0u, "run-begin", MAPS_NUM_TOKENS);
        run_tile_tokens(&plan, MAPS_NUM_TOKENS, &idma_ctrl, &eu_ctrl);
        maps_trace_event(&plan, 0u, 0u, "run-done", MAPS_NUM_TOKENS);
    }

    maps_main_trace(hartid, "barrier2");
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, MAPS_WAIT_MODE);
    maps_main_trace(hartid, "barrier2-done");

    if (hartid == 0u) {
        maps_main_trace(hartid, "check");
        uint32_t errors = maps_generated_check_output();
        printf("maps_handwritten errors: %d\n", errors);
        return errors;
    }

    return 0;
}
