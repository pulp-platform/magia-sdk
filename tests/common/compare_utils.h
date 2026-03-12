#ifndef COMPARE_UTILS_H_
#define COMPARE_UTILS_H_

#include <stdbool.h>
#include <stdint.h>

#include "tile.h"

/**
 * One ULP is the difference between two consecutive FP16 numbers
 * A tollerance of 8 ULP means that two values can differ by a maximum
 * of 8 consecutive representable FP16 values.
 */
#define ULP_TOLL    (80)

/**
 * @brief Check if a FP16 value is NaN or (+/-)Inf
 *
 * @param [in] x FP16 value represented as uint16_t
 *
 * In IEEE754 half precision:
 *   - exponent = 11111 (0x7C00) indicates special values
 *   - mantissa = 0   -> Inf
 *   - mantissa != 0  -> NaN
 *
 * @return true if NaN or Inf, false otherwise
 */
static inline bool fp16_is_invalid(uint16_t x)
{
    return (x & 0x7C00) == 0x7C00;
}

/**
 * @brief Convert a FP16 raw value into ordered I32
 *
 * @param [in] uint16_t x FP16 value casted to uitn16_t
 *
 * If negative, remove sign (i % 0x7FFF) and invert order (0x8000 - )
 * If positive, move the value to the "upper" part of I32 range (i + 0x8000)
 *
 * @return int32_t {description}
 */
static inline int32_t fp16_to_ordered(uint16_t x)
{
    int32_t i = (int32_t) x;
    return (i & 0x8000) ? (0x8000 - (i & 0x7FFF)) : (i + 0x8000);
}

/**
 * @brief Compare two FP16 vectors using Units in the Last Place tollerance
 *
 * @param [in] uintptr_t addr_res result vector start address
 * @param [in] uintptr_t addr_exp expected vector start address
 * @param [in] int len number of elements of the two vectors
 *
 * @return true if match, false otherwise
 */
static inline bool vector_compare_fp16_bitwise(uintptr_t addr_res, uintptr_t addr_exp, int len) {
    uint16_t expected;
    uint16_t result;
    int32_t ord_exp;
    int32_t ord_res;
    uint32_t offset;
    int32_t ulp_dif;
    bool ret;

    ret = true;
    for (int i = 0; i < len; i++) {
        offset = i * sizeof (uint16_t);

        expected = mmio16(addr_exp + offset);
        result = mmio16(addr_res + offset);

        /* Reject NaN or Inf */
        if (fp16_is_invalid(expected) || fp16_is_invalid(result)) {
            printf("Invalid FP16 value at idx %d\t-\texpected: %x\t-\tcomputed: %x\n", i, expected, result);
            ret = false;
            continue;
        }

        ord_exp = fp16_to_ordered(expected);
        ord_res = fp16_to_ordered(result);

        ulp_dif = (ord_exp > ord_res) ? (ord_exp - ord_res) : (ord_res - ord_exp);

        if (ulp_dif > ULP_TOLL) {
            printf("Mismatch at index %d\t-\texpected: %x\t-\tcomputed: %x\t-\tulp: %d\n", i, expected, result, ulp_dif);
            ret = false;
        }
    }

    return ret;
}

#endif  /* COMPARE_UTILS_H_ */
