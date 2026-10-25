#include <stdint.h>

#include "eventunit.h"
#include "tile.h"

#include "swiglu_fast_fp16_spatz.h"
#include "swiglu_fast_fp16_spatz_params.h"
#include "swiglu_fast_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "swiglu_fast_fp16_spatz"

void MAGIA_swiglu_fast_fp16_spatz_l1(float16 *Y, const float16 *G, const float16 *U, uint32_t len)
{
    volatile swiglu_fast_fp16_spatz_params_t params = {
        .shard_G = (uintptr_t)G,
        .shard_U = (uintptr_t)U,
        .shard_Y = (uintptr_t)Y,
        .len     = len,
    };
    int ret;

    spatz_run_task_with_params(SWIGLU_FAST_FP16_SPATZ_TASK, (uint32_t)&params);

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
