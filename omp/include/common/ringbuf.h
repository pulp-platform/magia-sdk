// Copyright 2024 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Noah Huetter <nhuetter@student.ethz.ch>
// Cyril Koenig <cykoenig@iis.ee.ethz.ch>

/* Description: Shared host/device ring buffer for software mailboxes.
 *
 * Producer-owned and consumer-owned fields live in separate cache-line-aligned
 * structs so concurrent updates never false-share a cache line. The same ABI
 * is used by the 64-bit host and 32-bit device builds.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

// TODO: Add assertion based on the device tree
#ifndef HERO_CACHELINE_BYTES
/* Conservative default: covers CVA6 (16 B lines) and typical 64 B lines. */
#define HERO_CACHELINE_BYTES 64
#endif

/**
 * Producer-owned state (and init-time metadata read by both sides).
 * Written by the producer: head
 * Written at init only: size, element_size, data_v, data_p
 */
struct ring_buf_producer {
  uint32_t head;
  uint32_t size;
  uint32_t element_size;
  uint32_t _reserved;
  uint64_t data_v; /* host virtual address of the payload buffer */
  uint64_t data_p; /* device-visible / physical address of the payload buffer */
} __attribute__((aligned(HERO_CACHELINE_BYTES)));

/**
 * Consumer-owned state.
 * Written by the consumer: tail
 */
struct ring_buf_consumer {
  uint32_t tail;
} __attribute__((aligned(HERO_CACHELINE_BYTES)));

/**
 * Ring buffer control block shared between host and device.
 * Must be allocated with alignment >= HERO_CACHELINE_BYTES for the nested
 * alignments to hold in absolute addresses (relative separation is still
 * guaranteed by the padded producer size).
 */
struct ring_buf {
  struct ring_buf_producer prod;
  struct ring_buf_consumer cons;
} __attribute__((aligned(HERO_CACHELINE_BYTES)));

_Static_assert(sizeof(struct ring_buf_producer) % HERO_CACHELINE_BYTES == 0,
               "producer block must span whole cache lines");
_Static_assert(sizeof(struct ring_buf_consumer) % HERO_CACHELINE_BYTES == 0,
               "consumer block must span whole cache lines");
_Static_assert(offsetof(struct ring_buf, cons) >= HERO_CACHELINE_BYTES,
               "producer and consumer must be on different cache lines");
_Static_assert(offsetof(struct ring_buf, cons) % HERO_CACHELINE_BYTES == 0,
               "consumer block must be cache-line aligned within ring_buf");

/**
 * @brief Init the ring buffer. See `struct ring_buf` for details.
 */
static inline void rb_init(volatile struct ring_buf *rb, uint32_t size,
                           uint32_t element_size) {
  rb->prod.head = 0;
  rb->cons.tail = 0;
  rb->prod.size = size;
  rb->prod.element_size = element_size;
}

/**
 * @brief Pop element from ring buffer using the host virtual data pointer.
 *
 * @return 0 on success, -1 if empty
 */
static inline int rb_host_get(volatile struct ring_buf *rb, void *el) {
  uint32_t tail = rb->cons.tail;
  uint32_t head = rb->prod.head;
  if (tail == head) {
    return -1;
  }
  for (uint32_t i = 0; i < rb->prod.element_size; i++) {
    *((uint8_t *)el + i) =
        *((uint8_t *)(uintptr_t)rb->prod.data_v + rb->prod.element_size * tail + i);
  }
  asm volatile("fence r,w");
  rb->cons.tail = (tail + 1) % rb->prod.size;
  return 0;
}

/**
 * @brief Push element into ring buffer using the host virtual data pointer.
 *
 * @return 0 on success, -1 if full
 */
static inline int rb_host_put(volatile struct ring_buf *rb, void *el) {
  uint32_t tail = rb->cons.tail;
  uint32_t head = rb->prod.head;
  uint32_t next_head = (head + 1) % rb->prod.size;
  if (next_head == tail) {
    return -1;
  }
  for (uint32_t i = 0; i < rb->prod.element_size; i++) {
    *((uint8_t *)(uintptr_t)rb->prod.data_v + rb->prod.element_size * head + i) =
        *((uint8_t *)el + i);
  }
  asm volatile("fence w,w");
  rb->prod.head = next_head;
  return 0;
}

/**
 * @brief Pop element from ring buffer using the device physical data pointer.
 *
 * @return 0 on success, -1 if empty
 */
static inline int rb_device_get(volatile struct ring_buf *rb, void *el) {
  uint32_t tail = rb->cons.tail;
  uint32_t head = rb->prod.head;
  if (tail == head) {
    return -1;
  }
  for (uint32_t i = 0; i < rb->prod.element_size; i++) {
    *((uint8_t *)el + i) =
        *((uint8_t *)(uintptr_t)rb->prod.data_p + rb->prod.element_size * tail + i);
  }
  asm volatile("fence r,w");
  rb->cons.tail = (tail + 1) % rb->prod.size;
  return 0;
}

/**
 * @brief Push element into ring buffer using the device physical data pointer.
 *
 * @return 0 on success, -1 if full
 */
static inline int rb_device_put(volatile struct ring_buf *rb, void *el) {
  uint32_t tail = rb->cons.tail;
  uint32_t head = rb->prod.head;
  uint32_t next_head = (head + 1) % rb->prod.size;
  if (next_head == tail) {
    return -1;
  }
  for (uint32_t i = 0; i < rb->prod.element_size; i++) {
    *((uint8_t *)(uintptr_t)rb->prod.data_p + rb->prod.element_size * head + i) =
        *((uint8_t *)el + i);
  }
  asm volatile("fence w,w");
  rb->prod.head = next_head;
  return 0;
}
