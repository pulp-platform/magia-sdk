// kernels_omp.c - the 3 regions of HERO2/apps/omp/basic/offload_benchmark/main.c,
// with real "#pragma omp target". The HERO clang generates their device code for
// the MAGIA cores (CV32E40P, rv32imc).
//
// The device side includes the HERO 64-bit API (MAGIA variant hero_64_magia.h),
// which provides the hero_load/store_uintXX used by OmpHostPointerLegalizer.
// NOTE: test_offload.c finds the kernels by the line of each #pragma omp target
// (_l28, _l32, _l36): do not add or remove lines above them.

#ifdef __HERO_DEV
#include <stdint.h>
#include "hero_64_magia.h"
#else
// TODO(RTEMS host): the host side is discarded today. To use regions() as the
// real host program, compile it with the RTEMS triple (see omp/hero_magia.cmake)
// and link it with libomptarget + plugin.

typedef unsigned int uint32_t;
#endif

#pragma omp begin declare target device_type(host)
int regions(void)
{
    uint32_t tmp_1 = 5;
    uint32_t tmp_2 = 10;

    // offload_benchmark: "enter_omp_init" / "enter_omp_nop"
#pragma omp target device(1)
    { asm volatile("nop"); }

    // offload_benchmark: "enter_omp_map_to"
#pragma omp target device(1) map(to : tmp_1)
    { volatile uint32_t local_tmp_1 = tmp_1; }

    // offload_benchmark: "enter_omp_map_to_from"
#pragma omp target device(1) map(tofrom : tmp_1, tmp_2)
    { tmp_1 = tmp_2; }

    return tmp_1 == tmp_2;
}
#pragma omp end declare target
