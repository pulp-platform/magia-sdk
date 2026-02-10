#include "magia_tile_utils.h"
#include "magia_spatz_utils.h"
#include "event_unit_utils.h"
#include "hello_task_bin.h"

int main(void) {

    int errors = 0;

    printf("[CV32] Spatz Test:\n");

    eu_init();
    eu_enable_events(EU_SPATZ_DONE_MASK);

    printf("\n[CV32] Initializing Spatz...\n");
    spatz_init(SPATZ_BINARY_START);

    printf("\n[CV32] Launching SPATZ Task\n");
    spatz_run_task(HELLO_TASK);

    printf("\n[CV32] Before eu wait\n");
    eu_wait_spatz_wfe(EU_SPATZ_DONE_MASK);

    if(spatz_get_exit_code() != 0) {
        printf("[CV32] SPATZ TASK ENDED with exit code: 0x%03x\n", spatz_get_exit_code());
        errors++;
    } else {
        printf("[CV32] SPATZ TASK ENDED successfully\n");
    }

    spatz_clk_dis();

    return errors;
}
