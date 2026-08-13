/*
 * L1 FIFO Communication Utils
 *
 * Lock-free multi-producer / single-consumer FIFO built from per-producer
 * circular queues (ring buffers) in L1 memory. The structure physically lives in
 * the *consumer* tile's L1; any other tile may push into it and the owning
 * consumer tile pops. Each producer is given its own independent circular
 * sub-ring, so the whole thing is genuinely circular (fixed slot arrays indexed
 * by monotonic head/tail counters that wrap modulo the capacity) rather than a
 * random-access slot pool.
 *
 * Why per-producer sub-rings: magia_v2 has no atomics ('a' extension), so a
 * single ring shared by many producers — which would need an atomic tail — is not
 * possible. Instead the consumer's FIFO is partitioned into P sub-rings, one per
 * producer, each a single-producer / single-consumer ring:
 *   - Only producer p writes sub-ring p's `tail`; only the consumer writes any
 *     `head`. Each counter therefore has exactly one writer -> no locking/CAS.
 *   - Producers key their sub-ring by a caller-chosen producer index. Using the
 *     producer's own hartid as that index (with num_producers = tile count) lets
 *     any tile push with no registration; a compact index also works when the
 *     producer set is known.
 *   - The consumer drains sub-rings round-robin (a scan cursor gives fairness /
 *     no starvation).
 *
 * Occupancy indicator: each sub-ring keeps free-running 32-bit `tail` (bumped by
 * its producer per push) and `head` (bumped by the consumer per pop) counters.
 * A sub-ring holds `tail - head` messages (unsigned, wrap-safe); the whole FIFO
 * holds the sum across sub-rings. fifo_count / fifo_count_from / fifo_is_full /
 * fifo_is_empty expose this without touching the payload.
 *
 * Ordering: a `fence w,w` before a producer bumps its `tail` guarantees the
 * consumer sees all payload/metadata before the slot becomes visible; a matching
 * `fence r,r` on the consumer orders the tail load ahead of the payload loads.
 * Backpressure: a producer whose sub-ring is full spins (reading the consumer's
 * `head`) until the consumer frees a slot, so it never overwrites unread data.
 *
 * Messages are self-describing: each slot carries a tensor_sub_slice_t descriptor
 * (the same type consumed by idma_memcpy_md_to_nd) describing the logical shape
 * and position of the payload, plus a caller-defined tag. The payload itself is
 * stored packed (row-major, contiguous) in the slot, making the FIFO usable for
 * arbitrary strided N-D tensor fragments, not just contiguous matrix rows.
 *
 * Memory layout at start of the consumer tile's L1:
 *   +0x00:  fifo_header_t          (16 bytes)
 *   +0x10:  fifo_ring_state_t[P]   (per-producer head/tail, 8 bytes each)
 *   +...:   slot array             (P * num_slots * slot_stride bytes)
 *           sub-ring p occupies slots [p*num_slots, (p+1)*num_slots)
 *
 * Each slot:
 *   +0x00: tag         (uint32_t) — caller-defined message tag
 *   +0x04: elem_bytes  (uint32_t) — element size in bytes
 *   +0x08: data_size   (uint32_t) — packed payload size in bytes
 *   +0x0C: desc        (tensor_sub_slice_t) — logical shape/position of payload
 *   +...:  payload data (slot_data_size bytes, inline, packed)
 */

#ifndef L1_FIFO_H
#define L1_FIFO_H

#include "magia_tile_utils.h"
#include "addr_map/tile_addr_map.h"
#include "idma.h"

typedef struct {
    uint32_t num_producers; /**< Number of per-producer sub-rings (P). */
    uint32_t num_slots;     /**< Slots per sub-ring (per-producer capacity). */
    uint32_t slot_stride;   /**< Bytes per slot (metadata + aligned payload). */
    uint32_t scan_cursor;   /**< Consumer round-robin cursor over producers. */
} fifo_header_t;

typedef struct {
    volatile uint32_t head; /**< Monotonic dequeue counter; consumer-owned. */
    volatile uint32_t tail; /**< Monotonic enqueue counter; producer-owned. */
} fifo_ring_state_t;

typedef struct {
    uint32_t tag;            /**< Caller-defined message tag. */
    uint32_t elem_bytes;     /**< Element size in bytes. */
    uint32_t data_size;      /**< Packed payload size in bytes (== desc.num_elems * elem_bytes). */
    tensor_sub_slice_t desc; /**< Logical shape/position of the payload. */
} fifo_slot_t;

/**
 * Message returned by fifo_peek / fifo_pop. Multi-value output boiled into one
 * struct. `data_ptr` points at the packed payload inside the slot (zero-copy);
 * the descriptor is copied out by value so it survives slot reuse. `src` is the
 * producer index the message came from — pass it to fifo_release.
 */
typedef struct {
    uint32_t data_ptr;       /**< Pointer to packed payload in the slot. */
    uint32_t src;            /**< Producer index this message came from. */
    uint32_t tag;            /**< Caller-defined message tag. */
    uint32_t elem_bytes;     /**< Element size in bytes. */
    uint32_t data_size;      /**< Payload size in bytes. */
    tensor_sub_slice_t desc; /**< Logical descriptor of the payload. */
} fifo_msg_t;

/**
 * Arguments for fifo_push, grouped into a struct.
 *
 * `src`/`src_base_addr` describe the producer-local (OBI) region to gather —
 * it may be strided. The payload is compacted into the sub-ring slot's packed
 * payload area. `desc` is the logical descriptor published to the consumer
 * (position via per-dim start, extent via length, byte strides of the reference
 * tensor); it shares its element count with `src`. The destination slot is chosen
 * internally (the tail slot of sub-ring `producer_idx`) — there is no
 * caller-assigned slot index.
 */
typedef struct {
    uint32_t target_hartid;        /**< Consumer tile whose FIFO is written. */
    uint32_t producer_idx;         /**< Which sub-ring to push into (this producer's). */
    uint32_t src_base_addr;        /**< Producer-local L1 base for the gather. */
    const tensor_sub_slice_t *src; /**< Producer-local layout (gather source). */
    const tensor_sub_slice_t *desc;/**< Logical descriptor to publish. */
    uint32_t tag;                  /**< Caller-defined message tag. */
    uint32_t elem_bytes;           /**< Element size in bytes. */
} fifo_push_req_t;

#define FIFO_HEADER_SIZE     (sizeof(fifo_header_t))
#define FIFO_RING_STATE_SIZE (sizeof(fifo_ring_state_t))
#define FIFO_SLOT_META_SIZE  (sizeof(fifo_slot_t))

/**
 * Get pointer to a tile's FIFO header in L1.
 */
static inline fifo_header_t *fifo_get_header(uint32_t target_hartid)
{
    return (fifo_header_t *)(L1_BASE + target_hartid * L1_TILE_OFFSET);
}

/**
 * Get the per-producer ring-state array (head/tail counters), right after header.
 */
static inline fifo_ring_state_t *fifo_ring_states(fifo_header_t *hdr)
{
    return (fifo_ring_state_t *)(hdr + 1);
}

/**
 * Get the ring-state (head/tail) for a given producer's sub-ring.
 */
static inline fifo_ring_state_t *fifo_ring_state(fifo_header_t *hdr, uint32_t producer_idx)
{
    return &fifo_ring_states(hdr)[producer_idx];
}

/**
 * Base of the slot array (after the header and the ring-state array).
 */
static inline uint8_t *fifo_slots_base(fifo_header_t *hdr)
{
    return (uint8_t *)fifo_ring_states(hdr) + hdr->num_producers * FIFO_RING_STATE_SIZE;
}

/**
 * Slot backing ring counter `counter` of producer `producer_idx` (wraps modulo
 * the per-sub-ring capacity).
 */
static inline fifo_slot_t *
fifo_slot_at(fifo_header_t *hdr, uint32_t producer_idx, uint32_t counter)
{
    uint32_t slot = producer_idx * hdr->num_slots + (counter % hdr->num_slots);
    return (fifo_slot_t *)(fifo_slots_base(hdr) + slot * hdr->slot_stride);
}

/**
 * Get pointer to a slot's inline (packed) payload data.
 */
static inline void *fifo_slot_data(fifo_slot_t *slot)
{
    return (void *)((uint8_t *)slot + FIFO_SLOT_META_SIZE);
}

/**
 * Initialize own tile's FIFO. Call once per tile at startup.
 *
 * num_producers:  number of per-producer sub-rings. Use the tile count and index
 *                 sub-rings by producer hartid to let any tile push; use a smaller
 *                 count with a compact producer index when the set is known. 0 for
 *                 tiles that only produce into another tile's FIFO.
 * num_slots:      slots per sub-ring (per-producer ring capacity).
 * slot_data_size: maximum packed payload size per slot in bytes.
 */
static inline void
fifo_init(uint32_t hartid, uint32_t num_producers, uint32_t num_slots, uint32_t slot_data_size)
{
    fifo_header_t *hdr         = fifo_get_header(hartid);
    uint32_t aligned_data_size = (slot_data_size + 3) & ~3u;

    hdr->num_producers = num_producers;
    hdr->num_slots     = num_slots;
    hdr->slot_stride   = FIFO_SLOT_META_SIZE + aligned_data_size;
    hdr->scan_cursor   = 0;

    fifo_ring_state_t *rs = fifo_ring_states(hdr);
    for (uint32_t p = 0; p < num_producers; p++) {
        rs[p].head = 0;
        rs[p].tail = 0;
    }

    asm volatile("fence w, w" ::: "memory");
}

/* ---- Occupancy / fill indicators (safe from either side) ------------------ */

/**
 * Messages queued in one producer's sub-ring: tail - head (unsigned, wrap-safe).
 */
static inline uint32_t fifo_count_from(uint32_t hartid, uint32_t producer_idx)
{
    fifo_ring_state_t *rs = fifo_ring_state(fifo_get_header(hartid), producer_idx);
    return rs->tail - rs->head;
}

/**
 * Total messages queued across all sub-rings.
 */
static inline uint32_t fifo_count(uint32_t hartid)
{
    fifo_header_t *hdr    = fifo_get_header(hartid);
    fifo_ring_state_t *rs = fifo_ring_states(hdr);
    uint32_t total        = 0;
    for (uint32_t p = 0; p < hdr->num_producers; p++)
        total += rs[p].tail - rs[p].head;
    return total;
}

/**
 * Per-sub-ring capacity (slots per producer).
 */
static inline uint32_t fifo_subring_capacity(uint32_t hartid)
{
    return fifo_get_header(hartid)->num_slots;
}

/**
 * Total capacity across all sub-rings.
 */
static inline uint32_t fifo_capacity(uint32_t hartid)
{
    fifo_header_t *hdr = fifo_get_header(hartid);
    return hdr->num_producers * hdr->num_slots;
}

/**
 * Non-zero if a given producer's sub-ring is full (a push would have to wait).
 */
static inline uint32_t fifo_producer_is_full(uint32_t hartid, uint32_t producer_idx)
{
    fifo_header_t *hdr    = fifo_get_header(hartid);
    fifo_ring_state_t *rs = fifo_ring_state(hdr, producer_idx);
    return (rs->tail - rs->head) >= hdr->num_slots;
}

/**
 * Non-zero if the FIFO holds no messages in any sub-ring.
 */
static inline uint32_t fifo_is_empty(uint32_t hartid)
{
    return fifo_count(hartid) == 0u;
}

/**
 * Non-zero if every sub-ring is full.
 */
static inline uint32_t fifo_is_full(uint32_t hartid)
{
    fifo_header_t *hdr    = fifo_get_header(hartid);
    fifo_ring_state_t *rs = fifo_ring_states(hdr);
    for (uint32_t p = 0; p < hdr->num_producers; p++)
        if ((rs[p].tail - rs[p].head) < hdr->num_slots)
            return 0;
    return 1;
}

/**
 * Build a packed (contiguous, row-major) descriptor with the same rank, per-dim
 * lengths and element count as `shape`. Per-dim start is 0 and byte strides are
 * derived from `elem_bytes` (innermost = elem_bytes, growing outward). Used to
 * describe the packed payload as laid out in a slot.
 */
static inline void
fifo_packed_slice(const tensor_sub_slice_t *shape, uint32_t elem_bytes, tensor_sub_slice_t *out)
{
    out->rank      = shape->rank;
    out->num_elems = shape->num_elems;

    uint32_t stride = elem_bytes;
    for (uint32_t d = shape->rank; d-- > 0u;) {
        out->dims[d].start  = 0u;
        out->dims[d].length = shape->dims[d].length;
        out->dims[d].stride = stride;
        stride *= shape->dims[d].length;
    }
}

/* ---- Producer side -------------------------------------------------------- */

/**
 * Reserve the next free slot of producer `producer_idx`'s sub-ring (its tail
 * slot), waiting via backpressure until that sub-ring has room. Does NOT advance
 * tail — call fifo_commit once the payload has been written. Returns a pointer to
 * the slot's packed payload area to write into.
 *
 * Single-producer-per-sub-ring: the tail slot is stable between reserve and commit.
 */
static inline void *fifo_reserve(uint32_t target_hartid, uint32_t producer_idx)
{
    fifo_header_t *hdr    = fifo_get_header(target_hartid);
    fifo_ring_state_t *rs = fifo_ring_state(hdr, producer_idx);

    /* Spin while this sub-ring is full; the consumer advances head as it drains. */
    while ((rs->tail - rs->head) >= hdr->num_slots)
        ;

    return fifo_slot_data(fifo_slot_at(hdr, producer_idx, rs->tail));
}

/**
 * Publish the reserved tail slot of producer `producer_idx`: store the descriptor
 * and metadata, fence, then advance that sub-ring's tail so the consumer sees a
 * fully-formed message. Pairs with fifo_reserve (or an external DMA that filled
 * the reserved slot's payload).
 */
static inline void fifo_commit(uint32_t target_hartid,
                               uint32_t producer_idx,
                               const tensor_sub_slice_t *desc,
                               uint32_t tag,
                               uint32_t elem_bytes)
{
    fifo_header_t *hdr    = fifo_get_header(target_hartid);
    fifo_ring_state_t *rs = fifo_ring_state(hdr, producer_idx);
    fifo_slot_t *slot     = fifo_slot_at(hdr, producer_idx, rs->tail);

    slot->desc       = *desc;
    slot->tag        = tag;
    slot->elem_bytes = elem_bytes;
    slot->data_size  = desc->num_elems * elem_bytes;

    /* Ensure all payload/metadata writes are visible before the tail bump. */
    asm volatile("fence w, w" ::: "memory");

    /* Publish: makes the entry visible to the consumer. */
    rs->tail = rs->tail + 1u;
}

/**
 * Push a (possibly strided) sub-slice into a target tile's FIFO, via this
 * producer's sub-ring (req->producer_idx).
 *
 * Reserves the sub-ring's tail slot (blocking on backpressure if it is full),
 * gathers the producer-local region described by req->src into the slot's packed
 * payload via idma_memcpy_md_to_nd (local L1 -> remote L1), waiting on each burst
 * through eu_ctrl (pass NULL in stalling mode), then commits (publishes) req->desc.
 *
 * Returns 0 on success, or the mover's error code (-1) on a bad descriptor. On
 * error the slot is left unpublished (tail is not advanced) and is reused by the
 * next push.
 */
static inline int fifo_push(idma_controller_t *idma_ctrl,
                            eu_controller_t *eu_ctrl,
                            const fifo_push_req_t *req)
{
    /* Guard before fifo_packed_slice indexes dims[rank-1] out of bounds. */
    if (req->src->rank > IDMA_ND_MAX_RANK)
        return -1;

    uint32_t dst = (uint32_t)fifo_reserve(req->target_hartid, req->producer_idx);

    tensor_sub_slice_t packed;
    fifo_packed_slice(req->src, req->elem_bytes, &packed);

    /* dir=1: OBI (local L1) -> AXI (remote tile L1 slot payload) */
    int rc = idma_memcpy_md_to_nd(
        idma_ctrl, 1, dst, req->src_base_addr, req->src, &packed, req->elem_bytes, eu_ctrl);
    if (rc != 0)
        return rc;

    fifo_commit(req->target_hartid, req->producer_idx, req->desc, req->tag, req->elem_bytes);
    return 0;
}

/**
 * CPU word-copy push for an already-packed source buffer (no DMA). Useful for
 * small control messages or tiles that do not drive the iDMA. `src_data` must
 * hold desc->num_elems * elem_bytes packed bytes. Blocks on backpressure if the
 * sub-ring is full.
 */
static inline void fifo_push_cpu(uint32_t target_hartid,
                                 uint32_t producer_idx,
                                 const void *src_data,
                                 const tensor_sub_slice_t *desc,
                                 uint32_t tag,
                                 uint32_t elem_bytes)
{
    void *dst           = fifo_reserve(target_hartid, producer_idx);
    uint32_t size_bytes = desc->num_elems * elem_bytes;

    volatile uint32_t *d = (volatile uint32_t *)dst;
    const uint32_t *s    = (const uint32_t *)src_data;
    uint32_t words       = size_bytes / 4;
    uint32_t remainder   = size_bytes % 4;

    for (uint32_t i = 0; i < words; i++)
        d[i] = s[i];

    if (remainder) {
        volatile uint8_t *db = (volatile uint8_t *)((uint8_t *)dst + words * 4);
        const uint8_t *sb    = (const uint8_t *)src_data + words * 4;

        for (uint32_t i = 0; i < remainder; i++)
            db[i] = sb[i];
    }

    fifo_commit(target_hartid, producer_idx, desc, tag, elem_bytes);
}

/* ---- Consumer side -------------------------------------------------------- */

/**
 * Look at the next ready message without consuming it. Scans sub-rings
 * round-robin from the header's scan cursor and returns the head message of the
 * first non-empty sub-ring (out->src identifies it). Returns 1 and fills *out if
 * a message is ready, else 0 (FIFO empty). Pure read: it does not advance any
 * head or the cursor, so repeated peeks are idempotent until fifo_release.
 *
 * The returned data_ptr points into the sub-ring's head slot and stays valid
 * until fifo_release(hartid, out->src) — so peek/verify/release is race-free even
 * while the producer keeps pushing (backpressure prevents the producer from
 * reusing the head slot before it is released).
 */
static inline uint32_t fifo_peek(uint32_t hartid, fifo_msg_t *out)
{
    fifo_header_t *hdr    = fifo_get_header(hartid);
    fifo_ring_state_t *rs = fifo_ring_states(hdr);
    uint32_t n            = hdr->num_producers;

    for (uint32_t i = 0; i < n; i++) {
        uint32_t p = (hdr->scan_cursor + i) % n;
        if (rs[p].tail == rs[p].head)
            continue;

        /* Order the tail observation ahead of the payload/metadata loads. */
        asm volatile("fence r, r" ::: "memory");

        fifo_slot_t *slot = fifo_slot_at(hdr, p, rs[p].head);
        out->data_ptr     = (uint32_t)fifo_slot_data(slot);
        out->src          = p;
        out->tag          = slot->tag;
        out->elem_bytes   = slot->elem_bytes;
        out->data_size    = slot->data_size;
        out->desc         = slot->desc;
        return 1;
    }

    return 0;
}

/**
 * Release the head slot of producer `producer_idx`'s sub-ring after a peeked
 * message has been fully consumed, advancing that head (so the producer may reuse
 * the slot) and the round-robin cursor. Pass the `src` returned by fifo_peek.
 * Call exactly once per successful fifo_peek.
 */
static inline void fifo_release(uint32_t hartid, uint32_t producer_idx)
{
    fifo_header_t *hdr    = fifo_get_header(hartid);
    fifo_ring_state_t *rs = fifo_ring_state(hdr, producer_idx);

    rs->head         = rs->head + 1u;
    hdr->scan_cursor = (producer_idx + 1u) % hdr->num_producers;
}

/**
 * Pop the next ready message: peek + immediate release. Returns 1 if a message
 * was dequeued, 0 if the FIFO was empty.
 *
 * Convenience for callers that copy out or fully consume the payload before any
 * further push to the same sub-ring can occur (e.g. a barrier-separated drain, or
 * sub-rings sized so slots are never reused). If the producer may reuse the head
 * slot concurrently, use fifo_peek/fifo_release so the payload is read before
 * release.
 */
static inline uint32_t fifo_pop(uint32_t hartid, fifo_msg_t *out)
{
    if (!fifo_peek(hartid, out))
        return 0;
    fifo_release(hartid, out->src);
    return 1;
}

#endif /* L1_FIFO_H */
