#include "tile.h"
#include "spmv_mac_params.h"

/* SPATZ task: dot product of one CSR row segment.
 * RES = sum_i( VAL[i] * X[i] )
 *
 * Done in float32, not int32: only the float vector path
 * (vfmacc.vv / vfredusum.vs / vfmv.f.s) is proven working on this
 * SPATZ build. VAL and X are prepared as float by the control core
 * (value and the sign-extended x, both cast to float); since CSR
 * values in this kernel are small quantized integers, the float32
 * dot product is exact and the control core casts the result back
 * to int32_t after the call.
 */
int spmv_mac_task(void)
{
    volatile spmv_mac_params_t *params;
    uintptr_t params_addr;
    int32_t *VAL;
    int32_t *X;
    volatile int32_t *RES;
    size_t avl;
    size_t vl;
    size_t vlmax;
    int32_t v2_dump[16];   /* big enough for your max vlmax */
    int32_t result;

    params_addr = mmio32(SPATZ_DATA);
    params      = (volatile spmv_mac_params_t *)params_addr;

    VAL = (int32_t *)params->chunk_VAL;
    X   = (int32_t *)params->chunk_X;
    RES = (volatile int32_t *)params->chunk_RES;

    avl = params->len;

    asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vlmax) : "r"(avl));
    asm volatile("vmv.v.i v2, 0");

    for (; avl > 0; avl -= vl) {
        asm volatile("vsetvli %0, %1, e32, m1, tu, ma" : "=r"(vl) : "r"(avl));
        asm volatile("vle32.v v0, (%0)" ::"r"(VAL));
        asm volatile("vle32.v v1, (%0)" ::"r"(X));
        asm volatile("vmacc.vv v2, v0, v1");
        VAL += vl;
        X += vl;
    }

    /* store v2 out instead of using vredsum/vmv.x.s */
    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(vlmax));
    /*
    asm volatile("vmv.v.i v3, 0");
    asm volatile("vredsum.vs v3, v2, v3");
    asm volatile("vmv.x.s %0, v3" : "=r"(result));
    */

    asm volatile("vse32.v v2, (%0)" ::"r"(v2_dump));

    result = 0;
    for (size_t i = 0; i < vlmax; i++) {
        result += v2_dump[i];
    }

    *RES = result;

    return 0;
}