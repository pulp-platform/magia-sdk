#ifndef KERNELS_COMPARE_UTILS_H_
#define KERNELS_COMPARE_UTILS_H_

#include <stdbool.h>
#include <stdint.h>

#include "tile.h"

#define HID get_hartid()

#define ULP_TOLL    (120)
// #define VERBOSE

static inline bool fp16_is_invalid(uint16_t x)
{
    return (x & 0x7C00) == 0x7C00;
}

static inline int32_t fp16_to_ordered(uint16_t x)
{
    int32_t i = (int32_t) x;
    return (i & 0x8000) ? (0x8000 - (i & 0x7FFF)) : (i + 0x8000);
}

static inline int compare_fp16_bitwise(uintptr_t addr_res, uintptr_t addr_exp, int len) {
    uint16_t expected;
    uint16_t result;
    int32_t ord_exp;
    int32_t ord_res;
    uint32_t offset;
    int32_t ulp_dif;
    int32_t ulp_avg;
    int n_mismatch;

    ulp_avg = 0;
    n_mismatch = 0;

    for (int i = 0; i < len; i++) {
        offset = i * sizeof (uint16_t);

        expected = mmio16(addr_exp + offset);
        result = mmio16(addr_res + offset);

        /* Reject NaN or Inf */
        if (fp16_is_invalid(expected) || fp16_is_invalid(result)) {
            printf("[CV32] Invalid FP16 value at idx %d\t-\texpected: %x\t-\tcomputed: %x\n", i, expected, result);
            n_mismatch++;
            continue;
        }

        ord_exp = fp16_to_ordered(expected);
        ord_res = fp16_to_ordered(result);

        ulp_dif = (ord_exp > ord_res) ? (ord_exp - ord_res) : (ord_res - ord_exp);
        ulp_avg += ulp_dif;

        if (ulp_dif > ULP_TOLL) {
            printf("[CV32] Mismatch at index %d\t-\texpected: %x\t-\tcomputed: %x\t-\tulp: %d\n", i, expected, result, ulp_dif);
            n_mismatch++;
        }
#ifdef VERBOSE
        else {
            printf("[CV32] Check on index %d SUCCESS\t-\texpected: %x\t-\tcomputed: %x\t-\tulp: %d\n", i, expected, result, ulp_dif);
        }
#endif
    }

    ulp_avg = ulp_avg / len;
    printf("[CV32] Number of mismatch: %u - Average ULP: %u\n", n_mismatch, ulp_avg);

    return n_mismatch;
}

#endif  /* KERNELS_COMPARE_UTILS_H_ */
