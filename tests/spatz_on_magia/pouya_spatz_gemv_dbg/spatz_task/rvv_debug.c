#include "rvv_debug.h"

#if RVV_DEBUG

static _Float16 dump0[256];
static _Float16 dump16[256];

void dump_v0(size_t vl)
{
    asm volatile("vse16.v v0, (%0)" :: "r"(dump0));

    printf("\nv0\n");

    for(size_t i=0;i<vl;i++)
        printf("[%3zu] %8.5f\n",i,(float)dump0[i]);
}

void dump_v16(size_t vl)
{
    asm volatile("vse16.v v16, (%0)" :: "r"(dump16));

    printf("\nv16\n");

    for(size_t i=0;i<vl;i++)
        printf("[%3zu] %8.5f\n",i,(float)dump16[i]);
}

#endif