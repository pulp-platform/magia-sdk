#include "tile.h"
#include "vmac_fs_params.h"

/* SPATZ task: dot product of two float32 vectors.
 * C = sum_i( A[i] * B[i] )
 * Runs a strip-mined vfmacc.vv loop (like the FFT example), then reduces
 * the partial-sum vector to a single scalar with vfredusum.vs.
 */
int vmac_fs_task(void)
{
    volatile vmac_fs_params_t *params;
    uintptr_t params_addr;
    float *A;
    float *B;
    volatile float *C;
    size_t avl;
    size_t vl;
    size_t vlmax;
    float result;

    params_addr = mmio32(SPATZ_DATA);
    params      = (volatile vmac_fs_params_t *)params_addr;

    A = (float *)params->chunk_A;
    B = (float *)params->chunk_B;
    C = (volatile float *)params->chunk_C;

    avl = params->len;

    /* Set vl to the max chunk width and zero the accumulator vector v2 */
    asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vlmax) : "r"(avl));
    asm volatile("vmv.v.i v2, 0");

    for (; avl > 0; avl -= vl) {
        asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"(avl));

        asm volatile("vle32.v v0, (%0)" ::"r"(A));
        asm volatile("vle32.v v1, (%0)" ::"r"(B));

        // v2 += v0 * v1 (multiply-accumulate)
        asm volatile("vfmacc.vv v2, v0, v1");

        A += vl;
        B += vl;
    }

    // Reduce the vlmax lanes of v2 down to a single scalar sum
    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(vlmax));
    asm volatile("vmv.v.i v3, 0");
    asm volatile("vfredusum.vs v3, v2, v3");
    asm volatile("vfmv.f.s %0, v3" : "=f"(result));

    *C = result;

    return 0;
}