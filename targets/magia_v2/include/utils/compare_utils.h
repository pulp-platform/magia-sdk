#ifndef COMPARE_UTILS_H_
#define COMPARE_UTILS_H_

#include <stdbool.h>
#include <math.h>

#include "magia_tile_utils.h"
#include "printf_wrapper.h"

static inline bool vector_compare_fp16_bitwise(uintptr_t addr_res, uintptr_t addr_exp, int len) {
    volatile uint16_t val_exp;
    volatile uint16_t val_res;
    uint32_t offset;
    bool ret;

    ret = true;
    for (int i = 0; i < len; i++) {
        offset = i * sizeof (uint16_t);

        val_exp = mmio16(addr_exp + offset);
        val_res = mmio16(addr_res + offset);

        if (val_exp != val_res) {
            printf("Mismatch at index %d - expected raw: 0x%04x - computed raw: 0x%04x\n", i, val_exp, val_res);
            ret = false;
        }
    }

    return ret;
}

#endif  /* COMPARE_UTILS_H_ */
