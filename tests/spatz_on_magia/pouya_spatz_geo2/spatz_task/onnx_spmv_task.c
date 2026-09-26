#include "tile.h"
#include "onnx_spmv_params.h"

/*
 * SPATZ side: result = sum_i ( val[i] * x[i] )   (int32)
 * Everything else (gathering, chunking, accumulating partials) is done
 * by the control core.
 */
int spmv_task(void)
{
    volatile spmv_params_t *params;
    const int32_t *val;
    const int32_t *xv;
    int32_t *res;
    size_t avl;
    size_t vl;
    int32_t sum;

    params = (volatile spmv_params_t *)mmio32(SPATZ_DATA);

    val = (const int32_t *)params->addr_val;
    xv  = (const int32_t *)params->addr_x;
    res = (int32_t *)params->addr_res;
    avl = params->len;

    asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(avl));

    asm volatile("vle32.v v8,  (%0)" ::"r"(val));
    asm volatile("vle32.v v16, (%0)" ::"r"(xv));

    /* products */
    asm volatile("vmul.vv v8, v8, v16");

    /* sum of all products */
    asm volatile("vmv.s.x v24, zero");
    asm volatile("vredsum.vs v24, v8, v24");
    asm volatile("vmv.x.s %0, v24" : "=r"(sum));

    *res = sum;

    return 0;
}