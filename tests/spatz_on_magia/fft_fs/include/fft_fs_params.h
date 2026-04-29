#ifndef FFT_FS_PARAMS_H_
#define FFT_FS_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t chunk_A;      /* Tile's first operand chunk       */
    uintptr_t chunk_B;      /* Tile's second operand chunk      */
    uintptr_t chunk_W;      /* Tile's twiddle factor chunk      */
    uintptr_t chunk_C;      /* Tile's output operand chunk      */
    uintptr_t chunk_G;      /* Tile's golden model chunk        */
    uint32_t start;         /* Tile's chunk global start index  */
    uint32_t end;           /* Tile's chunk global end index    */
    uint32_t len;           /* Tile's chunk len                 */
} fft_fs_params_t;

#endif  /* FFT_FS_PARAMS_H_ */
