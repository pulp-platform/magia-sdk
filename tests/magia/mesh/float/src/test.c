// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Alberto Dequino <alberto.dequino@unibo.it>

#include "tile.h"

/**
 * This test aims to verify the functionality of _Float16 operations.
 */
int main(void){
    uint32_t hartid = get_hartid();

    volatile float a =  0.235f;
    volatile float b =  -65504.0f;

    printf("%x\n", a);

    if(hartid==0){
        if(b < a)
            printf("%x is bigger than %x\n", a, b);
        else
            printf("%x is smaller than %x\n", a, b);
    }
    

    magia_return(hartid, 0);
    return 0;  
}