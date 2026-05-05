// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Alberto Dequino <alberto.dequino@unibo.it>

#include <stdint.h>
#include "tile.h"

/**
 * This test gently asks for each tile to say hello world.
 */
int main(void){
    uint32_t hartid = get_hartid();

    int test = NULL;

    volatile float16alt a = -12.0456f;
    volatile float16alt b = 4.0f;
    volatile float16alt c = (a + b);
    //int test_integer = -123;
    
    uint32_t *v = (uint32_t *) &c; 
    printf("Hello world! %#x\n", *v);
    // printf("Hello world! %f\n", c);
    //printf("Hello world! %d\n", test_integer);

    return 0;
}