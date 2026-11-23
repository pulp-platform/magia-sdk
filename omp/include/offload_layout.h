// offload_layout.h
//
// Host/device contract: fixed L2 addresses and message codes. Included by both
// the host (RTEMS on QEMU) and the device (omptarget_magia.c).
// Replaces, in HERO:
//   - the Carfield mailbox LETTER_0/1 registers (HERO2/sw/libhero/src/host/carfield/snitch_cluster.h)
//   - the MBOX_* codes                         (HERO2/sw/libhero/include/libhero/host/hero_api.h)
//
// MAGIA v4 memory map (gvsoc/pulp/pulp/chips/magia_v4/arch.py, magia_v4/link.ld):
//   L2           0xC000_0000 - 0xFFFF_FFFF (L2_SIZE = 1 GB by default)
//   code         0xCC00_0000 (instrram, 32 KB)    <- do not use
//   data/.bss    0xCC01_0000 (dataram)            <- do not use
//   TEST_END     0xCCFF_0000                      <- do not use
//   stdout       0xFFFF_0004                      <- do not use
// The offload structures are therefore placed at the start of L2.
// With UMA (QEMU + GVSoC) host and MAGIA see L2 at the same addresses.
// TODO(RTEMS host): check that the whole 0xC000_0000 - 0xC100_0000 range is shared.

#pragma once
#include <stdint.h>

// ---- "Ready" cell: replaces LETTER_0/1 --------------------------------------
// Shared memory may survive between two runs, hence:
//   host:   flag = 0, addresses = 0, fence w,w      <- invalidate stale values
//           init the mailboxes
//           write OFL_H2A_ADDR / OFL_A2H_ADDR, fence w,w
//           flag = OFL_READY_MAGIC                  <- LAST
//   device: wait for flag == OFL_READY_MAGIC, fence r,r
//           read OFL_H2A_ADDR / OFL_A2H_ADDR
//           fence r,w, flag = 0                     <- consume the flag
// TODO(RTEMS host): run the host sequence before MAGIA's fetch_enable.

#define OFL_READY_FLAG   0xC0000000u
#define OFL_H2A_ADDR     0xC0000004u   // address of the host -> device mailbox
#define OFL_A2H_ADDR     0xC0000008u   // address of the device -> host mailbox
#define OFL_READY_MAGIC  0xCAFEBABEu

// ---- Mailboxes (struct ring_buf from ringbuf.h, 64 B aligned) ---------------
#define OFL_H2A_RING     0xC0000100u
#define OFL_A2H_RING     0xC0000200u
#define OFL_H2A_DATA     0xC0001000u   // h2a payload (64 x 4 B)
#define OFL_A2H_DATA     0xC0001400u   // a2h payload (64 x 4 B)
#define OFL_RING_SLOTS   64u           // HERO uses 16; the fake host queues everything upfront

// ---- Heap for mapped data and argument buffers ------------------------------
#define OFL_HEAP_BASE    0xC0010000u
#define OFL_HEAP_END     0xC1000000u

// ---- Protocol: same as HERO (hero_api.h / sw_mailbox.h) ---------------------
#define MBOX_DEVICE_START  (0x02U)   // host->dev: START, kernel, argbuf, 0
#define MBOX_DEVICE_DONE   (0x04U)   // dev->host: DONE, cycles, dma, tot, dma, issue, compute
#define MBOX_DEVICE_STOP   (0x0FU)   // host->dev: leave the loop
#define MBOX_DEVICE_LOGLVL (0x10U)   // host->dev: LOGLVL, value (ignored)
#define MBOX_DONE_WORDS    7u        // words of a DONE reply, DONE included
