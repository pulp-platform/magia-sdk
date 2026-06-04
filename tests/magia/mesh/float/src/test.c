// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Alberto Dequino <alberto.dequino@unibo.it>

#include "tile.h"

static inline uint32_t get_raw(const float val)
{
    uint32_t raw;
    memcpy(&raw, &val, sizeof(raw));
    return raw;
}

/**
 * This test aims to verify the functionality of _Float16 operations.
 */
int main(void)
{
    uint32_t hartid = get_hartid();

    volatile float a = 0.235f;
    volatile float b = -4.04f;

    printf("a: %x\n", get_raw(a));
    printf("b: %x\n", get_raw(b));

    volatile float c = (a + b);

    printf("a + b: %x\n", get_raw(c));

    if (hartid == 0) {
        if (b < a)
            printf("%x is bigger than %x\n", get_raw(a), get_raw(b));
        else
            printf("%x is smaller than %x\n", get_raw(a), get_raw(b));
    }

    return 0;
}