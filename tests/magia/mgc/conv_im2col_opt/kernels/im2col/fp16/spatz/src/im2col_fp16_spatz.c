#include <stdint.h>

#include "eventunit.h"
#include "tile.h"

#include "im2col_fp16_spatz.h"
#include "im2col_fp16_spatz_params.h"
#include "im2col_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "im2col_fp16_spatz"

void MAGIA_im2col_fp16_spatz_l1(const im2col_fp16_spatz_params_t *params)
{
    int ret;

    spatz_run_task_with_params(IM2COL_FP16_SPATZ_TASK, (uint32_t)params);

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
