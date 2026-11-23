// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>

#include "common/ringbuf.h"

/***********************************************************************************
 * MACROS
 ***********************************************************************************/

#define MBOX_DEVICE_READY (0x01U)
#define MBOX_DEVICE_START (0x02U)
#define MBOX_DEVICE_BUSY (0x03U)
#define MBOX_DEVICE_DONE (0x04U)
#define MBOX_DEVICE_STOP (0x0FU)
#define MBOX_DEVICE_LOGLVL (0x10U)
#define MBOX_HOST_READY (0x1000U)
#define MBOX_HOST_DONE (0x3000U)

#define SYS_exit 60
#define SYS_write 64
#define SYS_read 63
#define SYS_wake 1235
#define SYS_cycle 1236

/***********************************************************************************
 * TYPES
 ***********************************************************************************/

struct mboxes {
    uint32_t a2h_ptr;
    uint32_t h2a_ptr;
};


/***********************************************************************************
 * DATA
 ***********************************************************************************/
extern volatile struct ring_buf *g_a2h_rb;
extern volatile struct ring_buf *g_a2h_mbox;
extern volatile struct ring_buf *g_h2a_mbox;
extern volatile struct mboxes   *g_mboxes;

/***********************************************************************************
 * INLINES
 ***********************************************************************************/

/**
 * @brief Holds physical addresses of the shared L3
 * @a2h_rb: accelerator to host ring buffer
 * @head: base of heap memory
 */
struct l3_layout {
    uint32_t a2h_rb;
    uint32_t a2h_mbox;
    uint32_t h2a_mbox;
    uint32_t heap;
};

/***********************************************************************************
 * PUBLICS
 ***********************************************************************************/
int syscall(uint64_t which, uint64_t arg0, uint64_t arg1, uint64_t arg2,
            uint64_t arg3, uint64_t arg4);

static __attribute__((optnone)) void csleep(uint32_t cycles) {
    for(uint32_t i = 0; i < cycles; i++) {
        ;
    }
}

void snrt_hero_exit(int code);
/**
 * @brief Blocking mailbox read access
 */
int mailbox_read(uint32_t *buffer, size_t n_words);
/**
 * @brief Non-Blocking mailbox read access. Return 1 on success, 0 on fail
 */
int mailbox_try_read(uint32_t *buffer);
/**
 * @brief Blocking mailbox write access
 */
int mailbox_write(uint32_t word);
