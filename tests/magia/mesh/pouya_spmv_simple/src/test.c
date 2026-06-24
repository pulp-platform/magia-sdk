/*
This code is a test for parallel sparse matrix-vector multiplication
(SpMV) using the Compressed Sparse Row (CSR) format. 
It is designed to run on a multi-core system, where each core computes
a portion of the output vector `y` based on the input matrix and vector `x`.
The code also includes performance measurement and error checking against
expected results.

Copyright (c) 2024, Pouya Shirindhahrakfard, It is free to use, but please cite the reference.
*/


#include <stdint.h>

#include "tile.h"
#include "test.h"

#define NUM_CORES 2

int main(void){

    perf_start();
    int start = perf_get_cycles();

    uint32_t hartid = get_hartid();

    // Only use first 2 cores
    if(hartid >= NUM_CORES){
        return 0;
    }

    // -----------------------------
    // 1. Compute row partition
    // -----------------------------
    uint32_t rows_per_core = M / NUM_CORES;

    uint32_t start_row = hartid * rows_per_core;
    uint32_t end_row   = start_row + rows_per_core;

    if(end_row > M) end_row = M;

    // -----------------------------
    // 2. CSR SpMV (parallel)
    // -----------------------------

    for(int i = start_row; i < end_row; i++){

        uint32_t sum = 0;

        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){

            uint16_t val = values[j];
            uint16_t col = col_idx[j];

            sum += val * x[col];
        }

        y[i] = (uint16_t)sum;
    }

    // -----------------------------
    // 3. Simple wait (temporary)
    // -----------------------------
    //for(volatile int d = 0; d < 10000; d++); // crude sync

    int end = perf_get_cycles();
    printf("\n Core[%d] Cycles: %d\n", hartid, end - start);

    // -----------------------------
    // 4. Check result (only core 0)
    // -----------------------------
    if(hartid == 0){

        int errors = 0;

        for(int i = 0; i < M; i++){
            if(y[i] != y_expected[i]){
                errors++;
            }
        }

        printf("Errors: %d\n", errors);
        return errors;
    }

    return 0;
}