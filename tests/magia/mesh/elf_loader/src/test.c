// SPDX-License-Identifier: Apache-2.0
//
// Magia binary loaded by ELF loader running on RTEMS guest (shared memfd = MAGIA L2).
//

#include <stdint.h>
#include "tile.h"

#define GUEST_READY       0x5EADBEEFu
#define SYNC_FROM_GVSOC   0x9ABCDEF0u
#define SYNC_TO_GVSOC     0xAAAAAAAAu
#define START_FROM_CORE_0 0xDEADBEEFu

/* Exchange buffer shared with the host MCU, placed by the linker in the
 * .uma_xbuf section at 0xC1000000 (= RTEMS 0xC1000000), see link.ld. */
__attribute__((section(".uma_xbuf"), aligned(64), used))
volatile uint32_t uma_xbuf[16];

__attribute__((section(".dataram"), aligned(64), used))
volatile uint32_t startFromCore0[16];

int main(void)
{
    if (get_hartid() == 0)
    {
        volatile uint32_t *xbuf = uma_xbuf;
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
        
        for(uint32_t coreIdx=0; coreIdx<sizeof(startFromCore0)/sizeof(uint32_t); coreIdx++)
        {
            startFromCore0[coreIdx]=START_FROM_CORE_0;
        }
        __asm__ volatile ("fence rw, rw" ::: "memory");
    }
    else
    {
        while(startFromCore0[get_hartid()]!=START_FROM_CORE_0);
            ;
        printf("[MAGIA] ELF LOADER TEST: Start from core 0 received, I'm tile %u\n", get_hartid());
    }

    return 0;
}
