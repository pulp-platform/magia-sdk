/*
 * Copyright (C) 2025 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * SPDX-License-Identifier: Apache-2.0
 *
 * L1 FIFO Communication Utils
 *
 * Per-tile FIFO mailbox in L1 memory, protected by an atomic spin-lock.
 * Any tile can push a message (arbitrary byte payload) to any other tile's
 * FIFO. The FIFO is a singly-linked list with bump allocation in the
 * receiver tile's L1.
 *
 * Memory layout at start of each tile's L1:
 *   +0x00: lock        (uint32_t) — 0=free, 1=held
 *   +0x04: head        (uint32_t) — pointer to first node (0 if empty)
 *   +0x08: tail        (uint32_t) — pointer to last node  (0 if empty)
 *   +0x0C: alloc_next  (uint32_t) — bump allocator next free byte
 *   +0x10: FIFO nodes + payload data
 */

#ifndef L1_FIFO_H
#define L1_FIFO_H

#include "magia_tile_utils.h"
#include "addr_map/tile_addr_map.h"

typedef struct {
    volatile uint32_t lock;
    volatile uint32_t head;
    volatile uint32_t tail;
    volatile uint32_t alloc_next;
} fifo_header_t;

typedef struct {
    volatile uint32_t next;
    volatile uint32_t data_ptr;
    volatile uint32_t data_size;
} fifo_node_t;

#define FIFO_HEADER_SIZE (sizeof(fifo_header_t))
#define FIFO_NODE_SIZE (sizeof(fifo_node_t))

/**
 * Get pointer to a tile's FIFO header in L1.
 */
static inline fifo_header_t *fifo_get_header(uint32_t target_hartid)
{
    return (fifo_header_t *)(L1_BASE + target_hartid * L1_TILE_OFFSET);
}

/**
 * Spin-lock acquire via amoswap.w (same pattern as amo_lock_naive).
 */
static inline void fifo_lock(fifo_header_t *hdr)
{
    volatile uint32_t addr   = (uint32_t)&hdr->lock;
    volatile uint32_t retval = 1;
    while (retval) {
        asm volatile("addi t0, %0, 0" ::"r"(addr) : "t0");
        asm volatile("addi t1, %0, 0" ::"r"(retval) : "t1");
        asm volatile("addi t2, %0, 0" ::"r"(retval) : "t2");
        asm volatile("amoswap.w t2, t1, (t0)" ::: "t2", "t1", "t0");
        asm volatile("mv %0, t2" : "=r"(retval)::"t2");
    }
}

/**
 * Release lock (write 0 via amoswap).
 */
static inline void fifo_unlock(fifo_header_t *hdr)
{
    volatile uint32_t addr = (uint32_t)&hdr->lock;
    volatile uint32_t zero = 0;
    asm volatile("addi t0, %0, 0" ::"r"(addr) : "t0");
    asm volatile("addi t1, %0, 0" ::"r"(zero) : "t1");
    asm volatile("addi t2, %0, 0" ::"r"(zero) : "t2");
    asm volatile("amoswap.w t2, t1, (t0)" ::: "t2", "t1", "t0");
}

/**
 * Initialize own tile's FIFO. Call once per tile at startup.
 */
static inline void fifo_init(uint32_t hartid)
{
    fifo_header_t *hdr = fifo_get_header(hartid);
    hdr->lock          = 0;
    hdr->head          = 0;
    hdr->tail          = 0;
    hdr->alloc_next    = (uint32_t)hdr + FIFO_HEADER_SIZE;
}

/**
 * Bump-allocate bytes in target tile's L1 (4-byte aligned).
 * Must be called while holding the target's FIFO lock.
 */
static inline uint32_t fifo_alloc(fifo_header_t *hdr, uint32_t size_bytes)
{
    uint32_t ptr = hdr->alloc_next;
    /* Align size up to 4 bytes */
    uint32_t aligned_size = (size_bytes + 3) & ~3u;
    hdr->alloc_next       = ptr + aligned_size;
    return ptr;
}

/**
 * Push data to a target tile's FIFO.
 * Locks the target FIFO, bump-allocates a node + payload in target L1,
 * copies data from src_data, links the node to the tail, then unlocks.
 */
static inline void fifo_push(uint32_t target_hartid, void *src_data, uint32_t size_bytes)
{
    fifo_header_t *hdr = fifo_get_header(target_hartid);

    fifo_lock(hdr);

    /* Allocate node */
    uint32_t node_addr = fifo_alloc(hdr, FIFO_NODE_SIZE);
    fifo_node_t *node  = (fifo_node_t *)node_addr;

    /* Allocate payload space */
    uint32_t payload_addr = fifo_alloc(hdr, size_bytes);

    /* Copy payload word-by-word via volatile writes */
    volatile uint32_t *dst = (volatile uint32_t *)payload_addr;
    uint32_t *src          = (uint32_t *)src_data;
    uint32_t words         = size_bytes / 4;
    uint32_t remainder     = size_bytes % 4;
    for (uint32_t i = 0; i < words; i++) {
        dst[i] = src[i];
    }
    /* Copy remaining bytes */
    if (remainder) {
        volatile uint8_t *dst_b = (volatile uint8_t *)(payload_addr + words * 4);
        uint8_t *src_b          = (uint8_t *)(src_data) + words * 4;
        for (uint32_t i = 0; i < remainder; i++) {
            dst_b[i] = src_b[i];
        }
    }

    /* Fill in node */
    node->next      = 0;
    node->data_ptr  = payload_addr;
    node->data_size = size_bytes;

    /* Link into FIFO */
    if (hdr->tail == 0) {
        /* Empty list */
        hdr->head = node_addr;
        hdr->tail = node_addr;
    } else {
        /* Append to tail */
        fifo_node_t *old_tail = (fifo_node_t *)(hdr->tail);
        old_tail->next        = node_addr;
        hdr->tail             = node_addr;
    }

    fifo_unlock(hdr);
}

/**
 * Pop from own tile's FIFO.
 * Returns 1 if a node was dequeued, 0 if empty.
 * On success, *out_data_ptr receives the L1 pointer to payload data,
 * and *out_size receives the payload size in bytes.
 * The payload pointer remains valid until fifo_init() resets the allocator.
 */
static inline uint32_t fifo_pop(uint32_t hartid, uint32_t *out_data_ptr, uint32_t *out_size)
{
    fifo_header_t *hdr = fifo_get_header(hartid);

    fifo_lock(hdr);

    if (hdr->head == 0) {
        fifo_unlock(hdr);
        return 0;
    }

    fifo_node_t *node = (fifo_node_t *)(hdr->head);
    *out_data_ptr     = node->data_ptr;
    *out_size         = node->data_size;

    /* Advance head */
    hdr->head = node->next;
    if (hdr->head == 0) {
        hdr->tail = 0;
    }

    fifo_unlock(hdr);
    return 1;
}

/**
 * Lock-free check: is the FIFO empty?
 */
static inline uint32_t fifo_is_empty(uint32_t hartid)
{
    fifo_header_t *hdr = fifo_get_header(hartid);
    return (hdr->head == 0) ? 1 : 0;
}

/**
 * Get the current bump allocator pointer for a tile's FIFO.
 * Useful for tests to know where user data buffers can start.
 */
static inline uint32_t fifo_get_alloc_next(uint32_t hartid)
{
    fifo_header_t *hdr = fifo_get_header(hartid);
    return hdr->alloc_next;
}

#endif /* L1_FIFO_H */
