#include "magia_tile_utils.h"
#include "magia_spatz_utils.h"
#include "event_unit_utils.h"
#include "hello_spatz_task_bin.h"

int main(void) {

    int errors = 0;

    printf("[CV32] Spatz Test:\n");

    // ==========================================
    // Initialization of Event Unit and Spatz
    // ==========================================

    // Initialize Event Unit
    eu_init();
    eu_enable_events(EU_SPATZ_DONE_MASK);  // Enable Spatz DONE event

    // Enable clk and initialize Spatz (bootrom jumps to _start, does full init)
    printf("\n[CV32] Initializing Spatz...\n");
    spatz_init(SPATZ_BINARY_START);

    // ==========================================
    // Test: Spatz Task
    // ==========================================
    printf("\n[CV32] Launching SPATZ Task\n");
    spatz_run_task(HELLO_WORLD_SIMPLE_TASK);

    eu_wait_spatz_wfe(EU_SPATZ_DONE_MASK);

    if(spatz_get_exit_code() != 0) {
        printf("[CV32] SPATZ TASK ENDED with exit code: 0x%03x\n", spatz_get_exit_code());
        errors++;
    } else {
        printf("[CV32] SPATZ TASK ENDED successfully\n");
    }

    // Disable Spatz clock
    spatz_clk_dis();

    // Return error status for CV32
    return errors;
}
