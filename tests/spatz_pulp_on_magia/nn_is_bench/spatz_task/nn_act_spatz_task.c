// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Spatz task 1 of the nn_is_bench block: scale/bias followed by hardswish over
// the tile's whole post-GEMM block.
//
//   p = z * scale + bias
//   p = p * clamp(p/6 + 1/2, 0, 1)
//
// fp16, LMUL=8, so one iteration covers VLEN/16*8 = 128 elements at VLEN=256.
// The instruction mix mirrors tests/spatz_on_magia/onnx_hardswish, which is
// already validated against the GVSoC Spatz model.

#include "tile.h"
#include "nn_is_params.h"

static inline _Float16 f16_from_bits(uint16_t bits)
{
    union {
        uint16_t u;
        _Float16 f;
    } c;
    c.u = bits;
    return c.f;
}

static void act_hardswish(const _Float16 *src, _Float16 *dst, size_t len, _Float16 scale,
                          _Float16 bias)
{
    register _Float16 ALPHA asm("fs0") = (_Float16)(1.0f / 6.0f);
    register _Float16 BETA asm("fs1")  = (_Float16)0.5f;
    register _Float16 ZERO asm("fs2")  = (_Float16)0.0f;
    register _Float16 ONE asm("fs3")   = (_Float16)1.0f;
    register _Float16 SCALE asm("fs4") = scale;
    register _Float16 BIAS asm("fs5")  = bias;

    const _Float16 *p_src = src;
    _Float16 *p_dst       = dst;
    size_t avl            = len;
    size_t vl;

    for (; avl > 0; avl -= vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        asm volatile("vle16.v v0, (%0)" ::"r"(p_src));

        /* affine pre-scaling */
        asm volatile("vfmul.vf v0, v0, %0" ::"f"(SCALE));
        asm volatile("vfadd.vf v0, v0, %0" ::"f"(BIAS));

        /* hardswish: v8 = clamp(v0/6 + 1/2, 0, 1); v0 = v0 * v8 */
        asm volatile("vfmul.vf v8, v0, %0" ::"f"(ALPHA));
        asm volatile("vfadd.vf v8, v8, %0" ::"f"(BETA));
        asm volatile("vfmin.vf v8, v8, %0" ::"f"(ONE));
        asm volatile("vfmax.vf v8, v8, %0" ::"f"(ZERO));
        asm volatile("vfmul.vv v0, v8, v0");

        asm volatile("vse16.v v0, (%0)" ::"r"(p_dst));

        p_src += vl;
        p_dst += vl;
    }
}

int nn_act_spatz_task(void)
{
    volatile nn_spatz_act_params_t *params;
    uintptr_t params_addr;

    params_addr = mmio32(SPATZ_DATA);
    params      = (volatile nn_spatz_act_params_t *)params_addr;

    act_hardswish((const _Float16 *)params->src,
                  (_Float16 *)params->dst,
                  (size_t)params->len,
                  f16_from_bits(params->scale),
                  f16_from_bits(params->bias));

    return 0;
}
