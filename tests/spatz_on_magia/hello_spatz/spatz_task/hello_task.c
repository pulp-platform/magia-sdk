#include "printf.h"
#include "performance_utils.h"

int hello_task(void) {
    uint32_t cycle_start,cycle_stop;
    cycle_start=perf_get_cycles();
    printf("[SNITCH] Hello World from Spatz!\n");
    cycle_stop=perf_get_cycles();
	printf("[SNITCH] %d\n", cycle_stop-cycle_start);
    return 0;
}
