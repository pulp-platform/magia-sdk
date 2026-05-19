#include "tile.h"
#include "eventunit.h"

#include "hello_pulp_task_bin.h"

int main(void) {
    eu_config_t eu_cfg;
    eu_controller_t eu_ctrl;
    int errors = 0;

    printf("[CV32] Hello PULP Test\n");

    eu_cfg.hartid = get_hartid();
    eu_ctrl.base  = NULL;
    eu_ctrl.cfg   = &eu_cfg;
    eu_ctrl.api   = &eu_api;

    eu_init(&eu_ctrl);
    eu_pulp_init(&eu_ctrl, 0);

    uint32_t pulp_core_mask = 0x91; /* one-hot bitmask: which PULP cores to run the task on */
    printf("[CV32] Initializing PULP cluster (binary @ 0x%08x)\n", PULP_BINARY_START);
    pulp_init(PULP_BINARY_START);

    printf("[CV32] Dispatching HELLO_TASK to PULP cluster (mask=0x%02x)\n", pulp_core_mask);
    pulp_run_task(HELLO_PULP_TASK, pulp_core_mask);

    eu_pulp_wait(&eu_ctrl, WFE);

    printf("[CV32] PULP cluster done\n");
    return errors;
}
