#include <stdint.h>

#include "eventunit.h"
#include "tile.h"

#include "expminus_rowsum_fp16_spatz.h"
#include "expminus_rowsum_fp16_spatz_params.h"
#include "expminus_rowsum_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "expminus_rowsum_fp16_spatz"

void MAGIA_expminus_rowsum_fp16_spatz_l1(const float16 *X,
                                         const float16 *M,
                                         float16 *Y,
                                         float16 *S,
                                         float16 *A,
                                         float16 s,
                                         uint32_t rows,
                                         uint32_t cols)
{
    expminus_rowsum_fp16_spatz_params_t params = {
        .shard_X = (uintptr_t)X,
        .shard_M = (uintptr_t)M,
        .shard_Y = (uintptr_t)Y,
        .shard_S = (uintptr_t)S,
        .shard_A = (uintptr_t)A,
    };
    int ret;

    expminus_rowsum_fp16_spatz_set_scale(&params, s, rows, cols);
    asm volatile("" ::: "memory");

    spatz_run_task_with_params(EXPMINUS_ROWSUM_FP16_SPATZ_TASK, (uint32_t)&params);

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
