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

    float test_float = -12.0456;
    
    printf("Hello world! %f\n", test_float);

    return 0;
}