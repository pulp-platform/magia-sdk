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
#define ULP_TOLL    (100)

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
    int32_t ulp_avg;
    bool ret;

    ulp_avg = 0;
    ret = true;
    for (int i = 0; i < len; i++) {
        offset = i * sizeof (uint16_t);

        expected = mmio16(addr_exp + offset);
        result = mmio16(addr_res + offset);

        /* Reject NaN or Inf */
        if (fp16_is_invalid(expected) || fp16_is_invalid(result)) {
            printf("[CV32] Invalid FP16 value at idx %d\t-\texpected: %x\t-\tcomputed: %x\n", i, expected, result);
            ret = false;
            continue;
        }

        ord_exp = fp16_to_ordered(expected);
        ord_res = fp16_to_ordered(result);

        ulp_dif = (ord_exp > ord_res) ? (ord_exp - ord_res) : (ord_res - ord_exp);
        ulp_avg += ulp_dif;

        if (ulp_dif > ULP_TOLL) {
            printf("[CV32] Mismatch at index %d\t-\texpected: %x\t-\tcomputed: %x\t-\tulp: %d\n", i, expected, result, ulp_dif);
            ret = false;
        }
    }

    ulp_avg = ulp_avg / len;
    printf("[CV32] Average ULP: %u\n", ulp_avg);

    return ret;
}

/**
 * @brief Compare two FP16 matrices using Units in the Last Place tollerance
 *
 * @param [in] uintptr_t addr_res result matrix start address
 * @param [in] uintptr_t addr_exp expected matrix start address
 * @param [in] int rows number of rows
 * @param [in] int cols number of cols
 *
 * @return true if match, false otherwise
 */
static inline bool matrix_compare_fp16_bitwise(uintptr_t addr_res, uintptr_t addr_exp, int rows, int cols) {
    uint16_t expected;
    uint16_t result;
    int32_t ord_exp;
    int32_t ord_res;
    uint32_t offset;
    int32_t ulp_dif;
    int32_t ulp_avg;
    bool ret;

    ulp_avg = 0;
    ret = true;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int i = r * cols + c;
            offset = i * sizeof(uint16_t);

            expected = mmio16(addr_exp + offset);
            result = mmio16(addr_res + offset);

            /* Reject NaN or Inf */
            if (fp16_is_invalid(expected) || fp16_is_invalid(result)) {
                printf("[CV32] Invalid FP16 value at (%d,%d)\t-\texpected: %x\t-\tcomputed: %x\n",
                       r, c, expected, result);
                ret = false;
                continue;
            }

            ord_exp = fp16_to_ordered(expected);
            ord_res = fp16_to_ordered(result);

            ulp_dif = (ord_exp > ord_res) ? (ord_exp - ord_res) : (ord_res - ord_exp);
            ulp_avg += ulp_dif;

            if (ulp_dif > ULP_TOLL) {
                printf("[CV32] Mismatch at (%d,%d)\t-\texpected: %x\t-\tcomputed: %x\t-\tulp: %d\n",
                       r, c, expected, result, ulp_dif);
                ret = false;
            }
        }
    }

    ulp_avg = ulp_avg / (rows * cols);
    printf("[CV32] Average ULP: %u\n", ulp_avg);

    return ret;
}

static inline bool tensor_compare_fp16_bitwise(uintptr_t addr_res, uintptr_t addr_exp, int N, int C, int H, int W)
{
    uint16_t expected;
    uint16_t result;
    int32_t ord_exp;
    int32_t ord_res;
    uint32_t offset;
    int32_t ulp_dif;
    int32_t ulp_avg;
    bool ret;
    int idx;

    idx = 0;
    ret = true;
    ulp_avg = 0;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++, idx++) {

                    offset = idx * sizeof(uint16_t);

                    expected = mmio16(addr_exp + offset);
                    result   = mmio16(addr_res + offset);

                    if (fp16_is_invalid(expected) || fp16_is_invalid(result)) {
                        printf("[CV32] Invalid FP16 at (%d,%d,%d,%d) - exp: %x - got: %x\n",
                            n,c,h,w, expected, result);
                        ret = false;
                        continue;
                    }

                    ord_exp = fp16_to_ordered(expected);
                    ord_res = fp16_to_ordered(result);

                    ulp_dif = (ord_exp > ord_res) ?
                            (ord_exp - ord_res) :
                            (ord_res - ord_exp);

                    ulp_avg += ulp_dif;

                    if (ulp_dif > ULP_TOLL) {
                        printf("[CV32] Mismatch at (%d,%d,%d,%d) - exp: %x - got: %x - ulp: %d\n",
                            n,c,h,w, expected, result, ulp_dif);
                        ret = false;
                    }
                }
            }
        }
    }

    ulp_avg /= (N*C*H*W);
    printf("[CV32] Average ULP: %d\n", ulp_avg);

    return ret;
}

#endif  /* COMPARE_UTILS_H_ */
