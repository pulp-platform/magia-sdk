#include <stdint.h>

#include "tile.h"
#include "test.h"

#define NUM_CORES 1

int main(void){

    perf_start();
    int start = perf_get_cycles();

    uint32_t hartid = get_hartid();

    // use cores
    if(hartid >= NUM_CORES){
        return 0;
    }

    // -----------------------------
    // 1. Compute row range
    // -----------------------------
    uint32_t rows_per_core = M / NUM_CORES;

    uint32_t start_row = hartid * rows_per_core;
    uint32_t end_row   = start_row + rows_per_core;

    // -----------------------------
    // 2. Matrix multiplication
    // -----------------------------

    for(int i = start_row; i < end_row; i++){
        for(int j = 0; j < K; j++){

            uint32_t sum = 0;

            for(int k = 0; k < N; k++){
                sum += A[i*N + k] * B[k*K + j];
            }

            C[i*K + j] = sum;
        }
    }

    int end = perf_get_cycles();
    printf("[tile %d]Cycles: %d\n", hartid, end - start);

    // -----------------------------
    // 3. Only core 0 checks result
    // -----------------------------
    if(hartid == 0){

        int errors = 0;

        for(int i = 0; i < M*K; i++){
            if(C[i] != C_expected[i]){
                errors++;
            }
        }

        printf("Errors: %d\n", errors);
        return errors;
    }

    return 0;
}