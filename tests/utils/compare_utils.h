#ifndef COMPARE_UTILS_H_
#define COMPARE_UTILS_H_

#include <stdbool.h>

#include "magia_tile_utils.h"

#define TOLL 0.004f

static inline bool vector_compare_fp16(uintptr_t addr_res, uintptr_t addr_exp, int len) {
    uint32_t offset;
    float abs_diff;
    float computed;
    float expected;
    bool ret;

    ret = true;
    for (int i = 0; i < len; i++) {
        offset = i * sizeof(float16);

        computed = (float)mmio_fp16(addr_res + offset);
        expected = (float)mmio_fp16(addr_exp + offset);

        abs_diff = fabs(computed - expected);

        if (abs_diff > TOLL) {
            printf("Mismatch at index %d\n", i);
            ret = false;
        }
    }

    return ret;
}

#endif  /* COMPARE_UTILS_H_ */
