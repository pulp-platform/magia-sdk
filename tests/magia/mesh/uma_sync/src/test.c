// SPDX-License-Identifier: Apache-2.0
//
// UMA handshake between MAGIA (GVSoC) and an RTEMS guest on QEMU.
//
// The QEMU guest RAM and the MAGIA L2 are the same memfd, and the MAGIA L2
// (0xC000_0000) is mapped on the part of it that QEMU sees at 0xC000_0000,
// so both sides use the same physical addresses. The exchange buffer is at a
// fixed position in the shared L2, independent of both applications
// (gvsoc pulp/chips/magia_v2/arch.py, UMA_EXCHANGE_ADDR): 0xC100_0000.
//
// RTEMS side (testsuites/samples/hello/init.c), exchange buffer [3]:
//     [0] syncFromGVSoC  RTEMS polls it for SYNC_FROM_GVSOC
//     [1] syncToGVSoC    RTEMS writes SYNC_TO_GVSOC after seeing it
//     [2] guestReady     RTEMS writes GUEST_READY once [0] and [1] are reset
//
// MAGIA side (hart 0 only): wait GUEST_READY -> write SYNC_FROM_GVSOC ->
// wait SYNC_TO_GVSOC -> done.

#include <stdint.h>
#include "tile.h"

/* Fixed exchange buffer, MAGIA L2 address (= RTEMS 0xC1000000) */
#define UMA_EXCHANGE_BUFFER 0xC1000000u

#define GUEST_READY     0x5EADBEEFu
#define SYNC_FROM_GVSOC 0x9ABCDEF0u
#define SYNC_TO_GVSOC   0xAAAAAAAAu

int main(void)
{
    if (get_hartid() != 0)
        return 0;

    volatile uint32_t *xbuf = (volatile uint32_t *)UMA_EXCHANGE_BUFFER;
    volatile uint32_t *sync_from_gvsoc = &xbuf[0];
    volatile uint32_t *sync_to_gvsoc   = &xbuf[1];
    volatile uint32_t *guest_ready     = &xbuf[2];

    printf("[MAGIA] exchange buffer at 0x%x (same address as RTEMS), waiting for guest\n",
           (uint32_t)(uintptr_t)xbuf);

    while (*guest_ready != GUEST_READY)
        ;
    __asm__ volatile ("fence rw, rw" ::: "memory");
    printf("[MAGIA] guest ready, sending 0x%x\n", SYNC_FROM_GVSOC);

    *sync_from_gvsoc = SYNC_FROM_GVSOC;
    __asm__ volatile ("fence rw, rw" ::: "memory");

    while (*sync_to_gvsoc != SYNC_TO_GVSOC)
        ;
    printf("[MAGIA] guest answered 0x%x, handshake done\n", *sync_to_gvsoc);

    return 0;
}
