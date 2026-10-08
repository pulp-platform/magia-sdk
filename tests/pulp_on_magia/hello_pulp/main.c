#include "tile.h"
#include "eventunit.h"

#include "hello_pulp_2_task_bin.h"

int main(void)
{
    int errors = 0;

    printf("[CV32] Hello PULP Test\n");

    printf("[CV32] Initializing Event Unit\n");
    eu_init();

    printf("[CV32] Initializing Pulp Event Unit\n");
    eu_pulp_init(0);

    uint32_t pulp_core_mask = 0x01; /* one-hot bitmask: which PULP cores to run the task on */
    printf("[CV32] Initializing PULP cluster (binary @ 0x%08x)\n", PULP_BINARY_START);
    pulp_init(PULP_BINARY_START);

    printf("[CV32] Dispatching HELLO_TASK to PULP cluster (mask=0x%02x)\n", pulp_core_mask);
    pulp_run_task(HELLO_PULP_TASK, pulp_core_mask);
    // pulp_run_task_with_params(HELLO_PULP_TASK, NULL, pulp_core_mask);

    eu_pulp_wait(WFE);

    printf("[CV32] PULP cluster done\n");
    return errors;
}
