/*
 * Copyright (C) 2023-2025 ETH Zurich and University of Bologna
 * Copyright (c) 1997-2010, 2012-2015 Wind River Systems, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Authors: Victor Isachi <victor.isachi@unibo.it>
 * Alberto Dequino <alberto.dequino@unibo.it>
 * 
 * PRF
 */
#ifndef PRF_H
#define PRF_H

#include <stdint.h>
#include <stdarg.h>
#include "io.h"

#ifndef MAXFLD
#define MAXFLD  200
#endif

/* Convention note: "end" as passed in is the standard "byte after
 * last character" style, but...
 */
static int reverse_and_pad(char *start, char *end, int minlen)
{
    int len;

    while (end - start < minlen) {
        *end++ = '0';
    }

    *end = 0;
    len = end - start;
    for (end--; end > start; end--, start++) {
        char tmp = *end;
        *end = *start;
        *start = tmp;
    }
    return len;
}

/* Writes the specified number into the buffer in the given base,
 * using the digit characters 0-9a-z (i.e. base>36 will start writing
 * odd bytes), padding with leading zeros up to the minimum length.
 */
static int to_x(char *buf, uint32_t n, int base, int minlen)
{
    char *buf0 = buf;

    do {
        int d = n % base;

        n /= base;
        *buf++ = '0' + d + (d > 9 ? ('a' - '0' - 10) : 0);
    } while (n);
    return reverse_and_pad(buf0, buf, minlen);
}

static int to_udec(char *buf, uint32_t value, int precision)
{
    return to_x(buf, value, 10, precision);
}

static int to_dec(char *buf, int32_t value, int fplus, int fspace, int precision)
{
    char *start = buf;

#if (MAXFLD < 10)
  return -1;
#endif

    if (value < 0) {
        *buf++ = '-';
        if (value != (int32_t)0x80000000)
            value = -value;
    } else if (fplus)
        *buf++ = '+';
    else if (fspace)
        *buf++ = ' ';

    return (buf + to_udec(buf, (uint32_t) value, precision)) - start;
}

/*
 *  The following two constants define the simulated binary floating
 *  point limit for the first stage of the conversion (fraction times
 *  power of two becomes fraction times power of 10), and the second
 *  stage (pulling the resulting decimal digits outs).
 */

#define MAXFP1  0xFFFFFFFF  /* Largest # if first fp format */
#define HIGHBIT64 (1ull<<63)

static int to_float(char *buf, uint64_t double_temp, int c,
                     int falt, int fplus, int fspace, int precision)
{
    register int    decexp;
    register int    exp;
    int             sign;
    int             digit_count;
    uint64_t        fract;
    uint64_t        ltemp;
    int             prune_zero;
    char           *start = buf;

    exp = double_temp >> 52 & 0x7ff;
    fract = (double_temp << 11) & ~HIGHBIT64;
    sign = !!(double_temp & HIGHBIT64);


    if (exp == 0x7ff) {
        if (sign) {
            *buf++ = '-';
        }
        if (!fract) {
            if (isupper(c)) {
                *buf++ = 'I';
                *buf++ = 'N';
                *buf++ = 'F';
            } else {
                *buf++ = 'i';
                *buf++ = 'n';
                *buf++ = 'f';
            }
        } else {
            if (isupper(c)) {
                *buf++ = 'N';
                *buf++ = 'A';
                *buf++ = 'N';
            } else {
                *buf++ = 'n';
                *buf++ = 'a';
                *buf++ = 'n';
            }
        }
        *buf = 0;
        return buf - start;
    }

    if (c == 'F') {
        c = 'f';
    }

    if ((exp | fract) != 0) {
        exp -= (1023 - 1);  /* +1 since .1 vs 1. */
        fract |= HIGHBIT64;
        decexp = 1;      /* Wasn't zero */
    } else
        decexp = 0;     /* It was zero */

    if (decexp && sign) {
        *buf++ = '-';
    } else if (fplus) {
        *buf++ = '+';
    } else if (fspace) {
        *buf++ = ' ';
    }

    decexp = 0;
    while (exp <= -3) {
        while ((fract >> 32) >= (MAXFP1 / 5)) {
            rlrshift(&fract);
            exp++;
        }
        fract *= 5;
        exp++;
        decexp--;

        while ((fract >> 32) <= (MAXFP1 / 2)) {
            fract <<= 1;
            exp--;
        }
    }

    while (exp > 0) {
        ldiv5(&fract);
        exp--;
        decexp++;
        while ((fract >> 32) <= (MAXFP1 / 2)) {
            fract <<= 1;
            exp--;
        }
    }

    while (exp < (0 + 4)) {
        rlrshift(&fract);
        exp++;
    }

    if (precision < 0)
        precision = 6;      /* Default precision if none given */
    prune_zero = 0;     /* Assume trailing 0's allowed     */
    if ((c == 'g') || (c == 'G')) {
        if (!falt && (precision > 0))
            prune_zero = 1;
        if ((decexp < (-4 + 1)) || (decexp > (precision + 1))) {
            if (c == 'g')
                c = 'e';
            else
                c = 'E';
        } else
            c = 'f';
    }

    if (c == 'f') {
        exp = precision + decexp;
        if (exp < 0)
            exp = 0;
    } else
        exp = precision + 1;
    digit_count = 16;
    if (exp > 16)
        exp = 16;

    ltemp = 0x0800000000000000;
    while (exp--) {
        ldiv5(&ltemp);
        rlrshift(&ltemp);
    }

    fract += ltemp;
    if ((fract >> 32) & 0xF0000000) {
        ldiv5(&fract);
        rlrshift(&fract);
        decexp++;
    }

    if (c == 'f') {
        if (decexp > 0) {
            while (decexp > 0) {
                *buf++ = get_digit(&fract, &digit_count);
                decexp--;
            }
        } else
            *buf++ = '0';
        if (falt || (precision > 0))
            *buf++ = '.';
        while (precision-- > 0) {
            if (decexp < 0) {
                *buf++ = '0';
                decexp++;
            } else
                *buf++ = get_digit(&fract, &digit_count);
        }
    } else {
        *buf = get_digit(&fract, &digit_count);
        if (*buf++ != '0')
            decexp--;
        if (falt || (precision > 0))
            *buf++ = '.';
        while (precision-- > 0)
            *buf++ = get_digit(&fract, &digit_count);
    }

    if (prune_zero) {
        while (*--buf == '0')
            ;
        if (*buf != '.')
            buf++;
    }

    if ((c == 'e') || (c == 'E')) {
        *buf++ = (char) c;
        if (decexp < 0) {
            decexp = -decexp;
            *buf++ = '-';
        } else
            *buf++ = '+';
        *buf++ = (char) ((decexp / 10) + '0');
        decexp %= 10;
        *buf++ = (char) (decexp + '0');
    }
    *buf = 0;

    return buf - start;
}

static int to_octal(char *buf, uint32_t value, int alt_form, int precision)
{
    char *buf0 = buf;

    if (alt_form) {
        *buf++ = '0';
        if (!value) {
            /* So we don't return "00" for a value == 0. */
            *buf++ = 0;
            return 1;
        }
    }
    return (buf - buf0) + to_x(buf, value, 8, precision);
}

static int to_hex(char *buf, uint32_t value,
           int alt_form, int precision, int prefix)
{
    int len;
    char *buf0 = buf;

    if (alt_form) {
        *buf++ = '0';
        *buf++ = 'x';
    }

    len = to_x(buf, value, 16, precision);
    if (prefix == 'X') {
        uc(buf0);
    }

    return len + (buf - buf0);
}

int prf(void (*func)(char), const char *format, va_list vargs)
{
    /*
     * Due the fact that buffer is passed to functions in this file,
     * they assume that it's size if MAXFLD + 1. In need of change
     * the buffer size, either MAXFLD should be changed or the change
     * has to be propagated across the file
     */
    char            buf[MAXFLD + 1];
    register int    c;
    int             count;
    register char   *cptr;
    int             falt;
    int             fminus;
    int             fplus;
    int             fspace;
    register int    i;
    int             need_justifying;
    char            pad;
    int             precision;
    int             prefix;
    int             width;
    char            *cptr_temp;
    int32_t         *int32ptr_temp;
    int32_t         int32_temp;
    uint32_t            uint32_temp;
    uint64_t            double_temp;

    count = 0;

    while ((c = *format++)) {
        if (c != '%') {
            (*func)(c);
            count++;
        } else {
            fminus = fplus = fspace = falt = 0;
            pad = ' ';      /* Default pad character    */
            precision = -1; /* No precision specified   */

            while (strchr("-+ #0", (c = *format++)) != NULL) {
                switch (c) {
                case '-':
                    fminus = 1;
                    break;

                case '+':
                    fplus = 1;
                    break;

                case ' ':
                    fspace = 1;
                    break;

                case '#':
                    falt = 1;
                    break;

                case '0':
                    pad = '0';
                    break;

                case '\0':
                    return count;
                }
            }

            if (c == '*') {
                /* Is the width a parameter? */
                width = (int32_t) va_arg(vargs, int32_t);
                if (width < 0) {
                    fminus = 1;
                    width = -width;
                }
                c = *format++;
            } else if (!isdigit(c))
                width = 0;
            else {
                width = atoi(&format); /* Find width */
                c = *format++;
            }

            /*
             * If <width> is INT_MIN, then its absolute value can
             * not be expressed as a positive number using 32-bit
             * two's complement.  To cover that case, cast it to
             * an unsigned before comparing it against MAXFLD.
             */
            if ((unsigned) width > MAXFLD) {
                width = MAXFLD;
            }

            if (c == '.') {
                c = *format++;
                if (c == '*') {
                    precision = (int32_t)
                    va_arg(vargs, int32_t);
                } else
                    precision = atoi(&format);

                if (precision > MAXFLD)
                    precision = -1;
                c = *format++;
            }

            /*
             * This implementation only checks that the following format
             * specifiers are followed by an appropriate type:
             *    h: short
             *    l: long
             *    L: long double
             *    z: size_t or ssize_t
             * No further special processing is done for them.
             */

            if (strchr("hlLz", c) != NULL) {
                i = c;
                c = *format++;
                /*
                 * Here there was a switch() block
                 * which was doing nothing useful, I
                 * am still puzzled at why it was left
                 * over. Maybe before it contained
                 * stuff that was needed, but in its
                 * current form, it was being
                 * optimized out.
                 */
            }

            need_justifying = 0;
            prefix = 0;
            switch (c) {
            case 'c':
                buf[0] = (char) ((int32_t) va_arg(vargs, int32_t));
                buf[1] = '\0';
                need_justifying = 1;
                c = 1;
                break;

            case 'd':
            case 'i':
                int32_temp = (int32_t) va_arg(vargs, int32_t);
                c = to_dec(buf, int32_temp, fplus, fspace, precision);
                if (fplus || fspace || (int32_temp < 0))
                    prefix = 1;
                need_justifying = 1;
                if (precision != -1)
                    pad = ' ';
                break;

            case 'e':
            case 'E':
            case 'f':
            case 'F':
            case 'g':
            case 'G':
                /* standard platforms which supports double */
            {
                union {
                    double d;
                    uint64_t i;
                } u;

                u.d = (double) va_arg(vargs, double);
                double_temp = u.i;
            }

                c = to_float(buf, double_temp, c, falt, fplus,
                          fspace, precision);
                if (fplus || fspace || (buf[0] == '-'))
                    prefix = 1;
                need_justifying = 1;
                break;

            case 'n':
                int32ptr_temp = (int32_t *)va_arg(vargs, int32_t *);
                *int32ptr_temp = count;
                break;

            case 'o':
                uint32_temp = (uint32_t) va_arg(vargs, uint32_t);
                c = to_octal(buf, uint32_temp, falt, precision);
                need_justifying = 1;
                if (precision != -1)
                    pad = ' ';
                break;

            case 'p':
                uint32_temp = (uint32_t) va_arg(vargs, uint32_t);
                c = to_hex(buf, uint32_temp, 1, 8, (int) 'x');
                need_justifying = 1;
                if (precision != -1)
                    pad = ' ';
                break;

            case 's':
                cptr_temp = (char *) va_arg(vargs, char *);
                /* Get the string length */
                for (c = 0; c < MAXFLD; c++) {
                    if (cptr_temp[c] == '\0') {
                        break;
                    }
                }
                if ((precision >= 0) && (precision < c))
                    c = precision;
                if (c > 0) {
                    memcpy(buf, cptr_temp, (size_t) c);
                    need_justifying = 1;
                }
                break;

            case 'u':
                uint32_temp = (uint32_t) va_arg(vargs, uint32_t);
                c = to_udec(buf, uint32_temp, precision);
                need_justifying = 1;
                if (precision != -1)
                    pad = ' ';
                break;

            case 'x':
            case 'X':
                uint32_temp = (uint32_t) va_arg(vargs, uint32_t);
                c = to_hex(buf, uint32_temp, falt, precision, c);
                if (falt)
                    prefix = 2;
                need_justifying = 1;
                if (precision != -1)
                    pad = ' ';
                break;

            case '%':
                (*func)('%');
                count++;
                break;

            case 0:
                return count;
            }

            if (c >= MAXFLD + 1)
                return -1;

            if (need_justifying) {
                if (c < width) {
                    if (fminus) {
                        /* Left justify? */
                        for (i = c; i < width; i++)
                            buf[i] = ' ';
                    } else {
                        /* Right justify */
                        (void) memmove((buf + (width - c)), buf, (size_t) (c
                                        + 1));
                        if (pad == ' ')
                            prefix = 0;
                        c = width - c + prefix;
                        for (i = prefix; i < c; i++)
                            buf[i] = pad;
                    }
                    c = width;
                }

                for (cptr = buf; c > 0; c--, cptr++, count++)
                    (*func)(*cptr);
            }
        }
    }
    return count;
}

#endif //PRF_H