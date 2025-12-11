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
 * Authors:
 * Alberto Dequino <alberto.dequino@unibo.it>
 * 
 * IO
 */
#ifndef IO_H
#define IO_H

#include <stdint.h>
#include <stddef.h>
#include "addr_map/tile_addr_map.h"


static void pputc(char c)
{
    *(volatile uint8_t *) (PRINT_ADDR) = (uint8_t)c;
}

char *strchr(const char *s, int c)
{
    char tmp = (char) c;

    while ((*s != tmp) && (*s != '\0'))
        s++;

    return (*s == tmp) ? (char *) s : NULL;
}

static inline int isdigit(int a)
{
	return (((unsigned)(a)-'0') < 10);
}

static int atoi(const char **sptr)
{
    register char *p;
    register int   i;

    i = 0;
    p = *sptr;
    p--;
    while (isdigit(((int) *p)))
        i = 10 * i + *p++ - '0';
    *sptr = p;
    return i;
}

static inline int isupper(int a)
{
	return ((unsigned)(a)-'A') < 26;
}

static  void rlrshift(uint64_t *v)
{
    *v = (*v & 1) + (*v >> 1);
}

/* Tiny integer divide-by-five routine.  The full 64 bit division
 * implementations in libgcc are very large on some architectures, and
 * currently nothing in Zephyr pulls it into the link.  So it makes
 * sense to define this much smaller special case here to avoid
 * including it just for printf.
 *
 * It works by iteratively dividing the most significant 32 bits of
 * the 64 bit value by 5.  This will leave a remainder of 0-4
 * (i.e. three significant bits), ensuring that the top 29 bits of the
 * remainder are zero for the next iteration.  Thus in the second
 * iteration only 35 significant bits remain, and in the third only
 * six.  This was tested exhaustively through the first ~10B values in
 * the input space, and for ~2e12 (4 hours runtime) random inputs
 * taken from the full 64 bit space.
 */
static void ldiv5(uint64_t *v)
{
    uint32_t i, hi;
    uint64_t rem = *v, quot = 0, q;
    static const char shifts[] = { 32, 3, 0 };

    /* Usage in this file wants rounded behavior, not truncation.  So add
     * two to get the threshold right.
     */
    rem += 2;

    for (i = 0; i < 3; i++) {
        hi = rem >> shifts[i];
        q = (uint64_t)(hi / 5) << shifts[i];
        rem -= q * 5;
        quot += q;
    }

    *v = quot;
}

static  char get_digit(uint64_t *fr, int *digit_count)
{
    int     rval;

    if (*digit_count > 0) {
        *digit_count -= 1;
        *fr = *fr * 10;
        rval = ((*fr >> 60) & 0xF) + '0';
        *fr &= 0x0FFFFFFFFFFFFFFFull;
    } else
        rval = '0';
    return (char) (rval);
}

static void uc(char *buf)
{
    for (/**/; *buf; buf++) {
        if (*buf >= 'a' && *buf <= 'z') {
            *buf += 'A' - 'a';
        }
    }
}

void *memcpy(void *dst0, const void *src0, size_t len0)
{
    void *save = dst0;

    char src_aligned = (((size_t) src0) & 3) == 0;
    char dst_aligned = (((size_t) dst0) & 3) == 0;
    char copy_full_words = (len0 & 3) == 0;

    if (src_aligned && dst_aligned && copy_full_words)
    {
        // all accesses are aligned => can copy full words
        uint32_t *dst = (uint32_t *) dst0;
        uint32_t *src = (uint32_t *) src0;

        while (len0)
        {
            *dst++ = *src++;
            len0 -= 4;
        }
    }
    else
    {
        // copy bytewise
        char *dst = (char *) dst0;
        char *src = (char *) src0;

        while (len0)
        {
            *dst++ = *src++;
            len0--;
        }
    }

    return save;
}

void *memmove(void *d, const void *s, size_t n)
{
    char *dest = d;
    const char *src  = s;

    if ((size_t) (dest - src) < n)
    {
        /*
         * The <src> buffer overlaps with the start of the <dest> buffer.
         * Copy backwards to prevent the premature corruption of <src>.
         */

        while (n > 0)
        {
            n--;
            dest[n] = src[n];
        }
    }
    else
    {
        /* It is safe to perform a forward-copy */
        while (n > 0)
        {
            *dest = *src;
            dest++;
            src++;
            n--;
        }
    }

    return d;
}


#endif // IO_H