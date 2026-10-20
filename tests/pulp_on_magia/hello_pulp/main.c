#include "tile.h"
#include "eventunit.h"

#include "hello_pulp_2_task_bin.h"

int main(void)
{
    int errors = 0;
    eu_init();

    printf("[CV32] Hello PULP Test\n");

    printf("[CV32] Initializing PULP cluster (binary @ 0x%08x)\n", PULP_BINARY_START);
    pulp_init(PULP_BINARY_START);

    printf("[CV32] Initializing Event Unit for accepting the cluster\n");
    eu_pulp_init(0);

    printf("[CV32] Dispatching HELLO_TASK to PULP cluster (core 0)\n");
    pulp_run_task(HELLO_PULP_TASK);

    eu_pulp_wait(WFE);

    printf("[CV32] PULP cluster done\n");
    return errors;
}
