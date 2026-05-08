#include "tile.h"
#include "fft_fs_params.h"

static inline uint16_t get_raw(const _Float16 val)
{
    uint16_t raw;
    memcpy(&raw, &val, sizeof(raw));
    return raw;
}
 
static inline void print_vector_raw(const _Float16 *vec, size_t len)
{
    for (size_t i = 0; i < len; i++) {
        printf("%d) %x\n", i, get_raw(vec[i]));
    }
}

int fft_fs_task(void)
{
    volatile fft_fs_params_t *params;
    uintptr_t params_addr;
    _Float16 *AR;
    _Float16 *AI;
    _Float16 *BR;
    _Float16 *BI;
    _Float16 *WR;
    _Float16 *WI;
    _Float16 *CR;
    _Float16 *CI;
    _Float16 *DR;
    _Float16 *DI;
    size_t avl;
    size_t vl;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile fft_fs_params_t *) params_addr;

    AR = (_Float16 *)params->chunk_AR;
    AI = (_Float16 *)params->chunk_AI;
    BR = (_Float16 *)params->chunk_BR;
    BI = (_Float16 *)params->chunk_BI;
    WR = (_Float16 *)params->chunk_WR;
    WI = (_Float16 *)params->chunk_WI;
    CR = (_Float16 *)params->chunk_CR;
    CI = (_Float16 *)params->chunk_CI;
    DR = (_Float16 *)params->chunk_DR;
    DI = (_Float16 *)params->chunk_DI;
    
    avl = params->len;

    for (; avl > 0; avl -= vl) {
        //TODO: lol
        asm volatile ("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(avl));

        // W*B -> (WR*BR - WI*BI) + i(BI*WR + WI*BR) = PR + iPI
        // A + WB -> (AR + PR) + i(AI + PI)
        // A - WB -> (AR - PR) + i(AI - PI)

        asm volatile ("vle16.v v0, (%0)" :: "r"(WR));
        asm volatile ("vle16.v v1, (%0)" :: "r"(WI));
        asm volatile ("vle16.v v2, (%0)" :: "r"(BR));
        asm volatile ("vle16.v v3, (%0)" :: "r"(BI));

        // PR = WR*BR - WI*BI (saved in v4)
        asm volatile ("vfmul.vv v4, v0, v2");
        asm volatile ("vfmul.vv v5, v1, v3");
        asm volatile ("vfsub.vv v4, v4, v5");

        // PI = BI*WR + WI*BR (saved in v5)
        asm volatile ("vfmul.vv v5, v3, v0");
        asm volatile ("vfmul.vv v6, v1, v2");
        asm volatile ("vfadd.vv v5, v5, v6");

        // Load A
        asm volatile ("vle16.v v2, (%0)" :: "r"(AR));
        asm volatile ("vle16.v v3, (%0)" :: "r"(AI));

        // CR = AR + PR
        asm volatile ("vfadd.vv v0, v2, v4");

        // CI = AI + PI
        asm volatile ("vfadd.vv v1, v3, v5");

        // Store C vector
        asm volatile ("vse16.v v0, (%0)" :: "r"(CR));
        asm volatile ("vse16.v v1, (%0)" :: "r"(CI));

        // DR = AR - PR
        asm volatile ("vfsub.vv v0, v2, v4");

        // DI = AI - PI
        asm volatile ("vfsub.vv v1, v3, v5");

        // Store D vector
        asm volatile ("vse16.v v0, (%0)" :: "r"(DR));
        asm volatile ("vse16.v v1, (%0)" :: "r"(DI));

        AR += vl;
        AI += vl;
        BR += vl;
        BI += vl;
        WR += vl;
        WI += vl;
        CR += vl;
        CI += vl;
        DR += vl;
        DI += vl;

    }

    // printf("AR VECTOR:\n");
    // print_vector_raw((const _Float16*) params->chunk_AR, (size_t) params->len);
    // printf("AI VECTOR:\n");
    // print_vector_raw((const _Float16*) params->chunk_AI, (size_t) params->len);
    // printf("BR VECTOR:\n");
    // print_vector_raw((const _Float16*) params->chunk_BR, (size_t) params->len);
    // printf("BI VECTOR:\n");
    // print_vector_raw((const _Float16*) params->chunk_BI, (size_t) params->len);

    // printf("WR VECTOR:\n");
    // print_vector_raw((const _Float16*) params->chunk_WR, (size_t) params->len);
    // printf("WI VECTOR:\n");
    // print_vector_raw((const _Float16*) params->chunk_WI, (size_t) params->len);

    // printf("CR VECTOR:\n");
    // print_vector_raw((const _Float16*) params->chunk_CR, (size_t) params->len);
    // printf("CI VECTOR:\n");
    // print_vector_raw((const _Float16*) params->chunk_CI, (size_t) params->len);
    // printf("DR VECTOR:\n");
    // print_vector_raw((const _Float16*) params->chunk_DR, (size_t) params->len);
    // printf("DI VECTOR:\n");
    // print_vector_raw((const _Float16*) params->chunk_DI, (size_t) params->len);

    return 0;
}
