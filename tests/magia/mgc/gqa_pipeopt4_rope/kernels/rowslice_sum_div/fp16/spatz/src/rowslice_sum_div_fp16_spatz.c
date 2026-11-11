#include <stdint.h>

#include "eventunit.h"
#include "tile.h"

#include "rowslice_sum_div_fp16_spatz.h"
#include "rowslice_sum_div_fp16_spatz_params.h"
#include "rowslice_sum_div_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "rowslice_sum_div_fp16_spatz"

void MAGIA_rowslice_sum_div_fp16_spatz_l1(float16 *Y,
                                          const float16 *O,
                                          const float16 *G,
                                          const float16 *S,
                                          uint32_t n,
                                          uint32_t own,
                                          uint32_t rows,
                                          uint32_t cols)
{
    volatile rowslice_sum_div_fp16_spatz_params_t params = {
        .shard_Y = (uintptr_t)Y,
        .shard_O = (uintptr_t)O,
        .shard_G = (uintptr_t)G,
        .shard_S = (uintptr_t)S,
        .dims    = rows | (cols << 16),
        .slices  = n | (own << 16),
    };
    int ret;

    spatz_run_task_with_params(ROWSLICE_SUM_DIV_FP16_SPATZ_TASK, (uint32_t)&params);

    ret = eu_spatz_wait(WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    ret = spatz_get_exit_code();
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
    }
}
