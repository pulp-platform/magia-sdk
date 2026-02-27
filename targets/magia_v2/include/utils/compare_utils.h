#ifndef COMPARE_UTILS_H_
#define COMPARE_UTILS_H_

#include <stdbool.h>
#include <math.h>

#include "magia_tile_utils.h"
#include "printf_wrapper.h"

#define TOLL 0x0011

static inline bool vector_compare_fp16_bitwise(uintptr_t addr_res, uintptr_t addr_exp, int len) {
    volatile uint16_t val_exp;
    volatile uint16_t val_res;
    volatile uint16_t abs_diff;
    uint32_t offset;
    bool ret;

    ret = true;
    for (int i = 0; i < len; i++) {
        offset = i * sizeof (uint16_t);

        val_exp = mmio16(addr_exp + offset);
        val_res = mmio16(addr_res + offset);
        abs_diff = (val_exp > val_res) ? (val_exp - val_res) : (val_res - val_exp);

        if (abs_diff > TOLL) {
            printf("Mismatch at index %d - expected raw: 0x%04x - computed raw: 0x%04x - abs_diff: 0x%04x\n", i, val_exp, val_res, abs_diff);
            ret = false;
        }
    }

    return ret;
}

#endif  /* COMPARE_UTILS_H_ */
