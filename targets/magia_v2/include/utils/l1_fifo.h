/*
 * L1 FIFO Communication Utils
 *
 * Lock-free per-tile FIFO mailbox in L1 memory.
 * Any tile can push a message to any other tile's FIFO by writing to a
 * pre-assigned slot. The owning tile polls slots for ready messages.
 *
 * Correctness relies on each (matrix_id, row_index) pair having exactly
 * one producer, so each slot is written by at most one remote tile.
 * A fence w,w before setting the valid flag ensures the consumer sees
 * all payload data before the slot becomes visible.
 *
 * Memory layout at start of each tile's L1:
 *   +0x00: fifo_header_t  (16 bytes)
 *   +0x10: slot array     (num_slots * slot_stride bytes)
 *
 * Each slot:
 *   +0x00: valid       (volatile uint32_t) — 0=empty, 1=ready
 *   +0x04: data_size   (uint32_t) — actual payload size in bytes
 *   +0x08: matrix_id   (uint32_t) — consumer-defined tag
 *   +0x0C: row_index   (uint32_t) — global row position
 *   +0x10: payload data (slot_data_size bytes, inline)
 */

#ifndef L1_FIFO_H
#define L1_FIFO_H

#include "magia_tile_utils.h"
#include "addr_map/tile_addr_map.h"

typedef struct {
    uint32_t num_slots;
    uint32_t slot_stride;
    uint32_t scan_cursor;
    uint32_t _pad;
} fifo_header_t;

typedef struct {
    volatile uint32_t valid;
    uint32_t data_size;
    uint32_t matrix_id;
    uint32_t row_index;
} fifo_slot_t;

#define FIFO_HEADER_SIZE    (sizeof(fifo_header_t))
#define FIFO_SLOT_META_SIZE (sizeof(fifo_slot_t))

/**
 * Get pointer to a tile's FIFO header in L1.
 */
static inline fifo_header_t *fifo_get_header(uint32_t target_hartid)
{
    return (fifo_header_t *)(L1_BASE + target_hartid * L1_TILE_OFFSET);
}

/**
 * Get pointer to slot at given index.
 */
static inline fifo_slot_t *fifo_get_slot(fifo_header_t *hdr, uint32_t slot_idx)
{
    return (fifo_slot_t *)((uint8_t *)(hdr + 1) + slot_idx * hdr->slot_stride);
}

/**
 * Get pointer to a slot's inline payload data.
 */
static inline void *fifo_slot_data(fifo_slot_t *slot)
{
    return (void *)((uint8_t *)slot + FIFO_SLOT_META_SIZE);
}

/**
 * Initialize own tile's FIFO. Call once per tile at startup.
 *
 * num_slots:      total number of message slots to pre-allocate
 * slot_data_size: maximum payload size per slot in bytes
 */
static inline void fifo_init(uint32_t hartid, uint32_t num_slots, uint32_t slot_data_size)
{
    fifo_header_t *hdr         = fifo_get_header(hartid);
    uint32_t aligned_data_size = (slot_data_size + 3) & ~3u;

    hdr->num_slots   = num_slots;
    hdr->slot_stride = FIFO_SLOT_META_SIZE + aligned_data_size;
    hdr->scan_cursor = 0;
    hdr->_pad        = 0;

    for (uint32_t i = 0; i < num_slots; i++) {
        fifo_slot_t *slot = fifo_get_slot(hdr, i);
        slot->valid       = 0;
    }

    asm volatile("fence w, w" ::: "memory");
}

/**
 * Push data to a target tile's FIFO at a specific slot.
 *
 * The caller must provide a slot_idx that is unique per (matrix_id, row_index)
 * pair for the target consumer. No locking is needed because each slot has
 * exactly one producer.
 */
static inline void fifo_push(uint32_t target_hartid,
                             uint32_t slot_idx,
                             void *src_data,
                             uint32_t size_bytes,
                             uint32_t matrix_id,
                             uint32_t row_index)
{
    fifo_header_t *hdr = fifo_get_header(target_hartid);
    fifo_slot_t *slot  = fifo_get_slot(hdr, slot_idx);
    void *dst          = fifo_slot_data(slot);

    /* Copy payload word-by-word via volatile writes */
    volatile uint32_t *d = (volatile uint32_t *)dst;

    uint32_t *s        = (uint32_t *)src_data;
    uint32_t words     = size_bytes / 4;
    uint32_t remainder = size_bytes % 4;

    for (uint32_t i = 0; i < words; i++)
        d[i] = s[i];

    if (remainder) {
        volatile uint8_t *db = (volatile uint8_t *)((uint8_t *)dst + words * 4);
        uint8_t *sb          = (uint8_t *)src_data + words * 4;

        for (uint32_t i = 0; i < remainder; i++)
            db[i] = sb[i];
    }

    /* Write metadata */
    slot->data_size = size_bytes;
    slot->matrix_id = matrix_id;
    slot->row_index = row_index;

    /* Ensure all data and metadata writes are visible before the valid flag */
    asm volatile("fence w, w" ::: "memory");

    /* Publish: makes the entry visible to the consumer */
    slot->valid = 1;
}

/**
 * Pop from own tile's FIFO.
 * Returns 1 if a ready slot was found, 0 if all slots are empty.
 * Scans round-robin from an internal cursor to avoid starvation.
 */
static inline uint32_t fifo_pop(uint32_t hartid,
                                uint32_t *out_data_ptr,
                                uint32_t *out_size,
                                uint32_t *out_matrix_id,
                                uint32_t *out_row_index)
{
    fifo_header_t *hdr = fifo_get_header(hartid);
    uint32_t n         = hdr->num_slots;

    if (n == 0)
        return 0;

    uint32_t cursor = hdr->scan_cursor;

    for (uint32_t count = 0; count < n; count++) {
        uint32_t idx      = (cursor + count) % n;
        fifo_slot_t *slot = fifo_get_slot(hdr, idx);

        if (slot->valid == 1) {
            *out_data_ptr  = (uint32_t)fifo_slot_data(slot);
            *out_size      = slot->data_size;
            *out_matrix_id = slot->matrix_id;
            *out_row_index = slot->row_index;

            /* Mark consumed */
            slot->valid = 0;

            /* Advance cursor past this slot */
            hdr->scan_cursor = (idx + 1) % n;
            return 1;
        }
    }

    return 0;
}

/**
 * Publish a slot after its payload has been written externally (e.g. via DMA).
 * Writes metadata, issues a fence, and sets the valid flag.
 */
static inline void fifo_slot_publish(uint32_t target_hartid,
                                     uint32_t slot_idx,
                                     uint32_t size_bytes,
                                     uint32_t matrix_id,
                                     uint32_t row_index)
{
    fifo_header_t *hdr = fifo_get_header(target_hartid);
    fifo_slot_t *slot  = fifo_get_slot(hdr, slot_idx);

    slot->data_size = size_bytes;
    slot->matrix_id = matrix_id;
    slot->row_index = row_index;

    asm volatile("fence w, w" ::: "memory");
    slot->valid = 1;
}

/**
 * Lock-free check: is the FIFO empty?
 */
static inline uint32_t fifo_is_empty(uint32_t hartid)
{
    fifo_header_t *hdr = fifo_get_header(hartid);
    for (uint32_t i = 0; i < hdr->num_slots; i++) {
        fifo_slot_t *slot = fifo_get_slot(hdr, i);
        if (slot->valid == 1)
            return 0;
    }
    return 1;
}

#endif /* L1_FIFO_H */
