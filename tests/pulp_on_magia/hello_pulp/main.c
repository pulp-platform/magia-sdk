#include "tile.h"
#include "eventunit.h"

#include "hello_pulp_bin.h"

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

    printf("[CV32] Starting PULP cluster (binary @ 0x%08x)\n", PULP_BINARY_START);
    pulp_init(PULP_BINARY_START);

    eu_pulp_wait(&eu_ctrl, WFE);

    printf("[CV32] PULP cluster done\n");
    return errors;
}
