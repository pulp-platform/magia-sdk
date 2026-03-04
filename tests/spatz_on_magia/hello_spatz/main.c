#include "tile.h"
#include "hello_spatz_task_bin.h"

#include "eventunit.h"

#define WAIT_MODE WFE

int main(void) {
    uint32_t hartid = get_hartid();

    eu_config_t eu_cfg = {.hartid = hartid};
    eu_controller_t eu_ctrl = {
        .base = NULL,
        .cfg = &eu_cfg,
        .api = &eu_api,
    };
    eu_init(&eu_ctrl);
    eu_spatz_init(&eu_ctrl, 0);

    int errors = 0;

    printf("[CV32] Spatz Test:\n");

    printf("[CV32] Initializing Spatz...\n");
    spatz_init(SPATZ_BINARY_START);

    printf("[CV32] Launching SPATZ Task\n");
    spatz_run_task(HELLO_TASK);

    eu_spatz_wait(&eu_ctrl, WAIT_MODE);

    if(spatz_get_exit_code() != 0) {
        printf("[CV32] SPATZ TASK ENDED with exit code: 0x%x\n", spatz_get_exit_code());
        errors++;
    } else {
        printf("[CV32] SPATZ TASK ENDED successfully\n");
    }

    spatz_clk_dis();

    return errors;
}
