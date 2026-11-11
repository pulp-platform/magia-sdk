#include <stdint.h>

#include "eventunit.h"
#include "tile.h"

#include "rowmax_fp16_spatz.h"
#include "rowmax_fp16_spatz_params.h"
#include "rowmax_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "rowmax_fp16_spatz"

void MAGIA_rowmax_fp16_spatz_l1(
    const float16 *X, const float16 *P, float16 *M, uint32_t rows, uint32_t cols)
{
    volatile rowmax_fp16_spatz_params_t params = {
        .shard_X = (uintptr_t)X,
        .shard_P = (uintptr_t)P,
        .shard_M = (uintptr_t)M,
        .dims    = rows | (cols << 16),
    };
    int ret;

    spatz_run_task_with_params(ROWMAX_FP16_SPATZ_TASK, (uint32_t)&params);

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
