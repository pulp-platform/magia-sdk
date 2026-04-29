#include "tile.h"
#include "fft_params.h"

int fft_fs_task(void)
{
    volatile fft_fs_params_t *params;
    uintptr_t params_addr;
    _Float16 *A;
    _Float16 *B;
    _Float16 *W;
    _Float16 *C;
    size_t avl;
    size_t vl;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile fft_fs_params_t *) params_addr;

    A = (_Float16 *)params->chunk_A;
    B = (_Float16 *)params->chunk_A;
    W = (_Float16 *)params->chunk_W;
    C = (_Float16 *)params->chunk_C;
    avl = params->len;

    for (; avl > 0; avl -= vl) {
        //TODO: lol
    }

    return 0;
}
