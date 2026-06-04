// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Victor Isachi <victor.isachi@unibo.it>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(){
    const unsigned int num_samples  = (unsigned int)pow(2, 14);
    const unsigned int max_rand     = 32768;
    const unsigned int num_per_line = 16;
    
    printf("// Auto-generated data array (uint32_t) of size %0u for Bitonic Sort\n\n", num_samples);
    printf("#ifndef _BS_ARRAY_%0u_\n#define _BS_ARRAY_%0u_\n\n", num_samples, num_samples);
    
    printf("#define BS_ARRAY_SIZE (%0u)\n\n", num_samples);
    printf("extern uint32_t bs_array [BS_ARRAY_SIZE] = {\n");
    int rand_num;
    for (unsigned int i = 0; i < num_samples-1; i++){
        rand_num = rand() % max_rand;
        printf("%5u,", rand_num);
        if ((i+1)%num_per_line) printf(" ");
        else                    printf("\n");
    }
    rand_num = rand() % max_rand;
    printf("%5u\n", rand_num);
    printf("};\n\n");
    
    printf("#endif /*_BS_ARRAY_%0u_*/", num_samples);

    return 0;
}