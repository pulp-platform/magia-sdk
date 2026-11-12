// SPDX-License-Identifier: Apache-2.0
//
// Magia binary loaded by ELF loader running on RTEMS guest (shared memfd = MAGIA L2).
//

#include <stdint.h>
#include "tile.h"

/* Fixed exchange buffer, MAGIA L2 address (= RTEMS 0xC1000000) */
#define UMA_EXCHANGE_BUFFER 0xC1000000u

#define GUEST_READY     0x5EADBEEFu
#define SYNC_FROM_GVSOC 0x9ABCDEF0u
#define SYNC_TO_GVSOC   0xAAAAAAAAu

int main(void)
{
    if (get_hartid() == 0)
    {
        volatile uint32_t *xbuf = (volatile uint32_t *)UMA_EXCHANGE_BUFFER;
        volatile uint32_t *sync_from_gvsoc = &xbuf[0];
        volatile uint32_t *sync_to_gvsoc   = &xbuf[1];
        volatile uint32_t *guest_ready     = &xbuf[2];

        printf("[MAGIA] ELF LOADER TEST: I'm tile %u\n", get_hartid());
        printf("[MAGIA] ELF LOADER TEST: exchange buffer at 0x%x (same address as RTEMS), waiting for guest\n",
               (uint32_t)(uintptr_t)xbuf);

        while (*guest_ready != GUEST_READY)
            ;
        __asm__ volatile ("fence rw, rw" ::: "memory");
        printf("[MAGIA] ELF LOADER TEST: guest ready, sending 0x%x\n", SYNC_FROM_GVSOC);

        *sync_from_gvsoc = SYNC_FROM_GVSOC;
        __asm__ volatile ("fence rw, rw" ::: "memory");

        while (*sync_to_gvsoc != SYNC_TO_GVSOC)
            ;
        printf("[MAGIA] ELF LOADER TEST: guest answered 0x%x, handshake done\n", *sync_to_gvsoc);
    }
    else
    {
        printf("[MAGIA] ELF LOADER TEST: I'm tile %u\n", get_hartid());
    }

    return 0;
}
