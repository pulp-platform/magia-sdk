#include "utils/magia_utils.h"
#include "addr_map/tile_addr_map.h"
#include "utils/printf.h"


void hello_pulp_task(void)
{
    uint32_t hartid   = get_hartid();
    uint32_t local_id = GET_PULP_LOCAL_ID(hartid);
    uint32_t tile_id  = GET_PULP_TILE_ID(hartid);
    //wait_nop((tile_id + 1) * hartid * 2000);
    printf("[PULP][tile-id=%u local-core-id=%u global-hartid=%u] Hello from PULP!\n",
           tile_id,
           local_id,
           hartid);
}
