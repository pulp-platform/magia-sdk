#ifndef FFT_FS_PARAMS_H_
#define FFT_FS_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t chunk_AR;      /* Tile's first operand chunk              */
    uintptr_t chunk_AI;      /* Tile's first operand chunk              */
    uintptr_t chunk_BR;      /* Tile's second operand chunk             */
    uintptr_t chunk_BI;      /* Tile's second operand chunk             */
    uintptr_t chunk_WR;      /* Tile's twiddle factor chunk             */
    uintptr_t chunk_WI;      /* Tile's twiddle factor chunk             */
    uintptr_t chunk_CR;      /* Tile's first output operand chunk       */
    uintptr_t chunk_CI;      /* Tile's first output operand chunk       */
    uintptr_t chunk_DR;      /* Tile's second output operand chunk      */
    uintptr_t chunk_DI;      /* Tile's second output operand chunk      */
    uint32_t len;            /* Tile's chunk len                        */
} fft_fs_params_t;

#endif  /* FFT_FS_PARAMS_H_ */
