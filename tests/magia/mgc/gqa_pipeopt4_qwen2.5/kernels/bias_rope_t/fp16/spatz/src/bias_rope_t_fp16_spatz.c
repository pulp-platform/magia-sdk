#include <stdint.h>

#include "eventunit.h"
#include "tile.h"

#include "bias_rope_t_fp16_spatz.h"
#include "bias_rope_t_fp16_spatz_params.h"
#include "bias_rope_t_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "bias_rope_t_fp16_spatz"

void MAGIA_bias_rope_t_fp16_spatz_l1(float16 *X,
                                     const float16 *B,
                                     const float16 *C,
                                     const float16 *S,
                                     uint32_t rows,
                                     uint32_t cols,
                                     uint32_t row0,
                                     uint32_t head_dim,
                                     uint32_t group)
{
    volatile bias_rope_t_fp16_spatz_params_t params = {
        .shard_X  = (uintptr_t)X,
        .bias     = (uintptr_t)B,
        .cos      = (uintptr_t)C,
        .sin      = (uintptr_t)S,
        .dims     = rows | (cols << 16),
        .row0     = row0,
        .head_dim = head_dim,
        .group    = group,
    };
    int ret;

    spatz_run_task_with_params(BIAS_ROPE_T_FP16_SPATZ_TASK, (uint32_t)&params);

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
