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

    float test_float = -12.0456f;
    //int test_integer = -123;
    
    printf("Hello world! %f\n", test_float);
    //printf("Hello world! %d\n", test_integer);

    return 0;
}