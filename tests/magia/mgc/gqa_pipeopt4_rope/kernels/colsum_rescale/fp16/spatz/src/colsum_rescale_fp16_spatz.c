#include <stdint.h>

#include "eventunit.h"
#include "tile.h"

#include "colsum_rescale_fp16_spatz.h"
#include "colsum_rescale_fp16_spatz_params.h"
#include "colsum_rescale_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "colsum_rescale_fp16_spatz"

void MAGIA_colsum_rescale_fp16_spatz_l1(
    float16 *S, const float16 *G, float16 *P, const float16 *A, uint32_t rows, uint32_t cols)
{
    volatile colsum_rescale_fp16_spatz_params_t params = {
        .shard_S = (uintptr_t)S,
        .shard_G = (uintptr_t)G,
        .shard_P = (uintptr_t)P,
        .shard_A = (uintptr_t)A,
        .dims    = rows | (cols << 16),
    };
    int ret;

    spatz_run_task_with_params(COLSUM_RESCALE_FP16_SPATZ_TASK, (uint32_t)&params);

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
