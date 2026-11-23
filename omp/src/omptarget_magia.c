// omptarget_magia.c - device-side OpenMP runtime for MAGIA v4 (1 tile).
//
// Minimal port of HERO2/sw/libhero/src/device/omptarget.c (main +
// gomp_offload_manager); sw_mailbox.c/.h are taken unchanged from HERO.
// Differences from HERO:
//   - core ID and cycles come from the magia-sdk (get_hartid, perf_init,
//     perf_get_cycles) instead of snitch_runtime.h;
//   - the mailbox addresses are read from the "ready" cell in L2
//     (offload_layout.h) instead of LETTER_0/1;
//   - shared variables are plain globals (placed in L2) instead of .noinit_l1;
//   - offloadArgs is loaded in full (HERO wrote only the low 32 bits);
//   - main() returns after STOP: crt0 writes TEST_END and GVSoC exits.

#include <stdint.h>
#include "tile.h"              // magia-sdk: get_hartid, perf_init, perf_get_cycles, printf
#include "common/ringbuf.h"
#include "sw_mailbox.h"
#include "offload_layout.h"

// Nothing to synchronize with 1 tile and 1 core.
// TODO: use the FractalSync barrier (fsync_sync_global(), hal/include/fsync.h)
// for multiple tiles.
static inline void magia_barrier(void)
{
}

// Shared between cores (.noinit_l1 in HERO)
static void (*volatile offloadFn)(uint64_t);
static volatile uint64_t offloadArgs;
static volatile uint32_t quit;

// Wait for START -> run the kernel -> reply DONE (HERO omptarget.c:48)
static int gomp_offload_manager(uint32_t core_idx)
{
    uint32_t cmd = 0, data = 0, word = 0;
    uint32_t cycles = 0;

    while (1) {
        if (core_idx == 0) {
            mailbox_read(&cmd, 1);
            if (cmd == MBOX_DEVICE_STOP) {
                quit = 1;
                magia_barrier();
                break;
            } else if (cmd == MBOX_DEVICE_LOGLVL) {
                mailbox_read(&data, 1);
                continue;
            } else if (cmd != MBOX_DEVICE_START) {
                printf("[magia] unknown command 0x%08x\n", cmd);
                break;
            }
            mailbox_read(&word, 1);  offloadFn   = (void (*)(uint64_t))(uintptr_t)word;
            mailbox_read(&word, 1);  offloadArgs = (uint64_t)word;
            mailbox_read(&data, 1);  // miss handler count, unused
            cycles = perf_get_cycles();
        }

        magia_barrier();
        if (quit)
            break;

        offloadFn(offloadArgs);

        magia_barrier();

        if (core_idx == 0) {
            cycles = perf_get_cycles() - cycles;
            mailbox_write(MBOX_DEVICE_DONE);
            mailbox_write(cycles);        // kernel cycles
            mailbox_write(0);             // DMA wait cycles (not measured)
            mailbox_write(0);             // perf: tot
            mailbox_write(0);             // perf: dma
            mailbox_write(0);             // perf: issue
            mailbox_write(0);             // perf: compute
        }
    }
    return 0;
}

// OFFLOAD_SELF_TEST: the core also acts as the host (fake host in test_offload.c).
// TODO(RTEMS host): on the real host this is done by libomptarget + plugin.
#ifdef OFFLOAD_SELF_TEST
void fake_host_before(void);
int  fake_host_after(void);
#endif

int main(void)
{
    uint32_t id = get_hartid();
    if (id != 0)
        return 0;                         // only tile 0 works

    perf_init();

    printf("[magia] entered omptarget main\n");

#ifdef OFFLOAD_SELF_TEST
    fake_host_before();                   // writes the mailboxes and the "ready" flag
#endif

    // Device side of the "ready" cell protocol (offload_layout.h)
    while (*(volatile uint32_t *)OFL_READY_FLAG != OFL_READY_MAGIC) { }
    asm volatile("fence r,r" ::: "memory");
    g_h2a_mbox = (volatile struct ring_buf *)(uintptr_t)*(volatile uint32_t *)OFL_H2A_ADDR;
    g_a2h_mbox = (volatile struct ring_buf *)(uintptr_t)*(volatile uint32_t *)OFL_A2H_ADDR;

    // Consume the flag after reading the addresses, so the next run does not see it valid
    asm volatile("fence r,w" ::: "memory");
    *(volatile uint32_t *)OFL_READY_FLAG = 0;

    printf("[magia] mailboxes found, entering gomp_offload_manager\n");
    gomp_offload_manager(0);
    printf("[magia] STOP received, exiting\n");

#ifdef OFFLOAD_SELF_TEST
    return fake_host_after();             // 0 = PASS
#else
    return 0;
#endif
}
