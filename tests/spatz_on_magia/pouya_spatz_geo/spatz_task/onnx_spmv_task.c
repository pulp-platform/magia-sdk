#include "tile.h"
#include "onnx_spmv_params.h"

/*
====================================================================
spmv_spatz_task

Deliberately the simplest possible kernel: the control core has
already built two dense, equal-length int32 arrays in L1 -- values[]
and x[] (x pre-gathered and sign-extended by the host) -- so SPATZ
just multiplies them element-wise and reduces to one scalar. No
indexed/gather loads, no strip-mining loop: the host already caps
len at SPATZ_MAX_VL (see mem_layout.h) and calls this task once per
chunk, accumulating partial sums itself across chunks/rows.

Register map:
    v8  = values chunk   (int32)
    v10 = x chunk        (int32, pre-gathered + sign-extended)
    v12 = v8 * v10        (elementwise product, int32)
    v14 = zero            (reduction seed)
    v15 = reduction result

NOTE [ASSUME]: the guide's reference kernel and "verified working"
instruction list (sec. 7 point 10) only cover the fp16 path
(vle16/vloxei16/vfmacc/vfredusum/vfmv). The integer instructions
used here (vle32.v, vmul.vv, vmv.v.x, vredsum.vs, vmv.x.s) are the
standard RVV 1.0 integer equivalents but are NOT on that verified
list for this Spatz build -- confirm each one in simulation before
relying on it, same as the guide recommends for any instruction
outside that list.
====================================================================
*/
int spmv_spatz_task(void)
{
    uintptr_t pa = mmio32(SPATZ_DATA);
    volatile spmv_spatz_params_t *p =
        (volatile spmv_spatz_params_t *)pa;

    const int32_t *values =
        (const int32_t *)p->addr_values;

    const int32_t *xvec =
        (const int32_t *)p->addr_xvec;

    volatile int32_t *result =
        (volatile int32_t *)p->addr_result;

    uint32_t n = p->len;

    int32_t sum;

    asm volatile(
        "vsetvli    t0, %[n], e32, m1, ta, ma  \n\t"
        "vle32.v    v8,  (%[val])              \n\t"
        "vle32.v    v10, (%[xv])                \n\t"
        "vmul.vv    v12, v8, v10                \n\t"
        "vmv.v.x    v14, x0                     \n\t"
        "vredsum.vs v15, v12, v14               \n\t"
        "vmv.x.s    %[sum], v15                 \n\t"
        : [sum] "=r" (sum)
        : [n]   "r" (n),
          [val] "r" (values),
          [xv]  "r" (xvec)
        : "t0", "v8", "v10", "v12", "v14", "v15", "memory"
    );

    *result = sum;

    return 0;
}