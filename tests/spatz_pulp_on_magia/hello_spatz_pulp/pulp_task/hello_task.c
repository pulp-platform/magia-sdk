#include "tile.h"

void hello_pulp(void)
{
    uint32_t hartid   = get_hartid();
    uint32_t local_id = cl_core_id();
    uint32_t tile_id  = cl_tile_id();
    // wait_nop((tile_id + 1) * hartid * 2000);
    printf("[PULP][tile-id=%u local-core-id=%u global-hartid=%u] Hello from PULP!\n",
           tile_id,
           local_id,
           hartid);
}

void hello_pulp_entry(void *)
{
    hello_pulp();
}

void hello_pulp_task(void)
{
    cl_eu_team_fork(PULP_CORE_COUNT, hello_pulp_entry, NULL);
    return;
}
