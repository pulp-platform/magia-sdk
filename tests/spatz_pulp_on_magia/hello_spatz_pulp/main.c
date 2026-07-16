#include "tile.h"
#include "eventunit.h"

#include "hello_spatz_task_bin.h" /* Spatz: SPATZ_BINARY_START, HELLO_TASK */
#include "hello_pulp_task_bin.h"  /* PULP:  PULP_BINARY_START              */

int main(void)
{
    eu_config_t eu_cfg;
    eu_controller_t eu_ctrl;
    int errors = 0;

    printf("[CV32] Hello Spatz+PULP Test\n");

    eu_cfg.hartid = get_hartid();
    eu_ctrl.base  = NULL;
    eu_ctrl.cfg   = &eu_cfg;
    eu_ctrl.api   = &eu_api;

    eu_init(&eu_ctrl);
    eu_pulp_init(&eu_ctrl, 0);
    eu_spatz_init(&eu_ctrl, 0);

    /* NOTE: spatz_init + pulp_init only load addresses / enable clocks;
     * pulp_run could be moved right after spatz_run_task to run both
     * clusters fully in parallel — no architectural issue with doing so. */

    printf("[CV32] Initializing Spatz\n");
    spatz_init(SPATZ_BINARY_START);

    uint32_t pulp_core_mask = 0x92; /* one-hot bitmask: which PULP cores to run the task on */
    printf("[CV32] Initializing PULP cluster (binary @ 0x%08x)\n", PULP_BINARY_START);
    pulp_init(PULP_BINARY_START);

    printf("[CV32] Launching Spatz task\n");
    spatz_run_task(HELLO_SPATZ_TASK);

    eu_spatz_wait(&eu_ctrl, WFE);
    if (spatz_get_exit_code() != 0) {
        printf("[CV32] Spatz FAILED (exit code 0x%03x)\n", spatz_get_exit_code());
        errors++;
    } else {
        printf("[CV32] Spatz done\n");
    }
    spatz_clk_dis();

    printf("[CV32] Dispatching HELLO_TASK to PULP cluster (mask=0x%02x)\n", pulp_core_mask);
    pulp_run_task(HELLO_PULP_TASK, pulp_core_mask);

    eu_pulp_wait(&eu_ctrl, WFE);
    printf("[CV32] PULP cluster done\n");

    return errors;
}
