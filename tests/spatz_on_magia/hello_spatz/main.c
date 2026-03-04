#include "tile.h"
#include "eventunit.h"

#include "hello_spatz_task_bin.h"

int main(void) {
    int errors;
    eu_config_t eu_cfg;
    eu_controller_t eu_ctrl;

    printf("[CV32] Hello Spatz Test\n");

    errors = 0;
    eu_cfg.hartid = get_hartid();
    eu_ctrl.base = NULL,
    eu_ctrl.cfg = &eu_cfg,
    eu_ctrl.api = &eu_api,

    printf("[CV32] Initializing Event Unit\n");
    eu_init(&eu_ctrl);

    printf("[CV32] Initializing Spatz Event Unit\n");
    eu_spatz_init(&eu_ctrl, 0);

    printf("[CV32] Initializing Spatz\n");
    spatz_init(SPATZ_BINARY_START);

    printf("[CV32] Launching SPATZ Task\n");
    spatz_run_task(HELLO_TASK);

    eu_spatz_wait(&eu_ctrl, WFE);

    if(spatz_get_exit_code() != 0) {
        printf("[CV32] SPATZ TASK ENDED with exit code: 0x%03x\n", spatz_get_exit_code());
        errors++;
    } else {
        printf("[CV32] SPATZ TASK ENDED successfully\n");
    }

    spatz_clk_dis();

    return errors;
}
