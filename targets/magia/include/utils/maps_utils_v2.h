#ifndef MAPS_UTILS_V2_H
#define MAPS_UTILS_V2_H

/*
 * FIFO-backed MAPS transport.
 *
 * This intentionally leaves maps_utils.h unchanged.  The compute, slice and
 * L2 descriptor types are reused verbatim; only the point-to-point transport
 * descriptors and tile plan are versioned here.  Keeping the common fields in
 * the same order as their v1 counterparts keeps generator changes mechanical.
 *
 * Receives whose packed payload already matches a whole, dense destination
 * slice are consumed in place (MAPS_FIFO_INPLACE_RECV, default on): the slice
 * is aliased to the FIFO slot for the token's operations and the slot is
 * released afterwards, instead of being unpacked into planned storage.
 *
 * Slices assembled from several receives, or laid out with padding, are
 * gathered in place instead: maps_fifo_init gives them a shared payload ring
 * whose slots mirror the slice's planned layout, and every sender writes its
 * piece straight to its final position there.  Once all pieces of a token
 * have arrived, the slice is aliased to that slot exactly as above.
 */

#include "utils/maps_utils.h"
#include "utils/maps_operations.h"
#include "utils/l1_fifo.h"
#include "utils/printf.h"

typedef struct {
    uint32_t transition_id;
    uint32_t src_hartid;
    uint32_t dst_hartid;
    subslice_desc_t dst;
    uint32_t producer_idx; /* Compact, destination-local FIFO sub-ring index. */
} fifo_recv_desc_t;

typedef struct {
    uint32_t transition_id;
    uint32_t src_hartid;
    uint32_t dst_hartid;
    subslice_desc_t src;
    subslice_desc_t dst;
    uint32_t dst_l1_data_base;          /* Retained for generator compatibility. */
    uint32_t dst_slice_l1_offset_bytes; /* Retained for generator compatibility. */
    uint32_t dst_slice_slot_bytes;      /* Retained for generator compatibility. */
    tensor_sub_slice_t copy_src;
    tensor_sub_slice_t copy_dst;
    uint32_t producer_idx;
    /* FIFO_PAYLOAD_* chosen by the generator; for FIFO_PAYLOAD_SHARED the piece
     * lands at dst_offset_bytes in the shared slot, laid out as copy_dst. */
    uint32_t dst_payload_mode;
    uint32_t dst_offset_bytes;
} fifo_send_desc_t;

typedef struct {
    uint32_t l1_offset_bytes;
    uint32_t num_producers;
    uint32_t num_slots;
    const uint32_t *slot_data_sizes;
    uint32_t slot_data_size;
    /* Shared payload rings decided by the generator: payload_rings[p] is the
     * 1-based ring producer p gathers into, or 0; NULL means none. */
    const uint32_t *payload_rings;
    uint32_t num_shared_rings;
    const uint32_t *shared_ring_bytes;
} fifo_desc_t;

typedef struct fifo_tile_plan {
    uint32_t hartid;

    uint32_t l1_data_base;
    uint32_t ready_flags_base;  /* Retained for compatible generated layouts. */
    uint32_t ready_flags_count; /* Unused by FIFO transport. */
    uint32_t num_token_slots;

    uint32_t num_slices;
    const slice_desc_t *slices;
    uint32_t num_init_l2_reads;
    const l2_read_desc_t *init_l2_reads;
    uint32_t num_l2_reads;
    const l2_read_desc_t *l2_reads;

    uint32_t num_recvs;
    const fifo_recv_desc_t *recvs;
    uint32_t num_ops;
    const op_desc_t *ops;
    uint32_t num_sends;
    const fifo_send_desc_t *sends;
    uint32_t num_l2_writes;
    const l2_write_desc_t *l2_writes;

    maps_operation_runtime_t *operation_runtime;
    fifo_desc_t fifo;
    /* Tile-local L1 offset the descriptor arrays are copied to at start-up
     * (maps_fifo_plan_to_l1); 0 keeps them in L2. */
    uint32_t descriptor_l1_offset;
} fifo_tile_plan_t;

typedef struct maps_fifo_pending_send {
    fifo_pending_push_t push;
    struct maps_fifo_pending_send *next;
    /* Older and newer unpublished pushes on the same consumer sub-ring: tail
     * advances at publication, so they must be published in this order. */
    struct maps_fifo_pending_send *prev_same;
    struct maps_fifo_pending_send *next_same;
    uint32_t token;
    uint32_t slot;
    uint32_t transition_id;
    uint32_t start_cycle;
} maps_fifo_pending_send_t;

typedef struct {
    maps_fifo_pending_send_t *sends;
    uint32_t num_sends;
} maps_fifo_token_state_t;

/* The MAGIA v3 RTL job FIFO has 16 entries.  GVSoC may apply backpressure
 * earlier, which is handled by the retry path below. */
#define MAPS_FIFO_MAX_PENDING_SENDS 16u

#ifdef MAPS_EXPERIMENT_TRACE
#define MAPS_EXPERIMENT_MAX_TOKENS 64u
typedef struct {
    uint32_t starts[NUM_HARTS][MAPS_EXPERIMENT_MAX_TOKENS];
    uint32_t ends[NUM_HARTS][MAPS_EXPERIMENT_MAX_TOKENS];
} maps_fifo_experiment_trace_t;

static inline maps_fifo_experiment_trace_t *maps_fifo_experiment_trace(void)
{
    static maps_fifo_experiment_trace_t trace
        __attribute__((section(".l2_bulk.maps_experiment_trace")));
    return &trace;
}
#endif

static inline uint32_t maps_fifo_tag(uint32_t transition_id, uint32_t slot)
{
    /* Preserve the old ready-flag namespace: transition_id * 16 + slot. */
    return transition_id * 16u + slot;
}

#if MAPS_FIFO_INPLACE_RECV
/* How a receive's payload reaches its destination slice. */
enum {
    MAPS_FIFO_RECV_UNPACK = 0u,  /* Copied into planned storage. */
    MAPS_FIFO_RECV_INPLACE = 1u, /* Whole dense slice, read from its own slot. */
    MAPS_FIFO_RECV_GATHER = 2u,  /* Piece of a slice in a shared payload ring. */
};

static inline uint32_t maps_fifo_op_uses_slice(const op_desc_t *op, uint32_t slice_id)
{
    for (uint32_t i = 0; i < op->num_inputs; ++i)
        if (op->inputs[i].slice_id == slice_id)
            return 1u;
    for (uint32_t i = 0; i < op->num_outputs; ++i)
        if (op->outputs[i].slice_id == slice_id)
            return 1u;
    return 0u;
}

/* A slice may live in FIFO memory for a token when no L2 read fills it and only
 * the token's operations read it.  The slots are released right after the
 * operations, before any transfer that could block on backpressure. */
static inline uint32_t maps_fifo_slice_inplace_ok(const fifo_tile_plan_t *plan,
                                                  uint32_t slice_id)
{
    for (uint32_t i = 0; i < plan->num_l2_reads; ++i)
        if (plan->l2_reads[i].dst.slice_id == slice_id)
            return 0u;

    /* Transfers out of the slice run after the release (and asynchronous
     * sends outlive the token); collective peers address this tile's slices
     * from the plan rather than through the alias. */
    for (uint32_t i = 0; i < plan->num_sends; ++i)
        if (plan->sends[i].src.slice_id == slice_id)
            return 0u;
    for (uint32_t i = 0; i < plan->num_l2_writes; ++i)
        if (plan->l2_writes[i].src.slice_id == slice_id)
            return 0u;
    for (uint32_t i = 0; i < plan->num_ops; ++i) {
        const op_desc_t *op = &plan->ops[i];
        if ((op->kind == OP_ALL_REDUCE_MAX || op->kind == OP_ALL_REDUCE_SUM ||
             op->collective.num_participants != 0u) &&
            maps_fifo_op_uses_slice(op, slice_id))
            return 0u;
    }
    return 1u;
}

/* A receive held in place must own its producer sub-ring: a second receive on
 * the same sub-ring would be hidden behind the held head slot. */
static inline uint32_t maps_fifo_recv_owns_subring(const fifo_tile_plan_t *plan,
                                                   uint32_t recv_idx)
{
    for (uint32_t i = 0; i < plan->num_recvs; ++i)
        if (i != recv_idx && plan->recvs[i].producer_idx == plan->recvs[recv_idx].producer_idx)
            return 0u;
    return 1u;
}

/* A receive may be consumed straight from its own FIFO slot when its packed
 * payload is byte-identical to the destination slice, nothing else writes the
 * slice, and the slice is otherwise eligible. */
static inline uint32_t maps_fifo_recv_inplace_ok(const fifo_tile_plan_t *plan,
                                                 uint32_t recv_idx)
{
    const fifo_recv_desc_t *recv = &plan->recvs[recv_idx];
    const subslice_desc_t *dst = &recv->dst;
    const slice_desc_t *slice = get_slice((const tile_plan_t *)plan, dst->slice_id);

    if (slice == NULL || slice->global_kind == GLOBAL_INITIALIZER)
        return 0u;
    if (dst->rank != slice->rank || dst->rank == 0u ||
        dst->elem_bytes != slice->elem_bytes)
        return 0u;

    /* The receive must cover the whole slice, which must be densely packed. */
    uint32_t packed_stride = dst->elem_bytes;
    for (uint32_t d = dst->rank; d-- > 0u;) {
        if (dst->offset[d] != 0u || dst->shape[d] != slice->shape[d] ||
            dst->strides_bytes[d] != slice->strides_bytes[d] ||
            slice->strides_bytes[d] != packed_stride)
            return 0u;
        packed_stride *= dst->shape[d];
    }

    for (uint32_t i = 0; i < plan->num_recvs; ++i)
        if (i != recv_idx && plan->recvs[i].dst.slice_id == dst->slice_id)
            return 0u;
    return maps_fifo_recv_owns_subring(plan, recv_idx) &&
           maps_fifo_slice_inplace_ok(plan, dst->slice_id);
}

/* Decide once per run how each receive is consumed.  Gathered slices come
 * from the generated FIFO layout (read back from the local FIFO header); they
 * take their aliases first, whole dense receives get the rest. */
static inline void maps_fifo_plan_inplace(const fifo_tile_plan_t *plan, uint8_t *inplace)
{
    uint32_t num_aliases = 0u;

    for (uint32_t i = 0; i < plan->num_recvs; ++i) {
        inplace[i] = MAPS_FIFO_RECV_UNPACK;
        if (!fifo_producer_is_shared(plan->hartid, plan->recvs[i].producer_idx))
            continue;
        inplace[i] = MAPS_FIFO_RECV_GATHER;
        uint32_t first = 1u;
        for (uint32_t j = 0; j < i; ++j)
            first &= plan->recvs[j].dst.slice_id != plan->recvs[i].dst.slice_id;
        num_aliases += first;
    }
    for (uint32_t i = 0; i < plan->num_recvs; ++i) {
        if (inplace[i] != MAPS_FIFO_RECV_UNPACK || num_aliases >= MAPS_MAX_SLICE_ALIASES ||
            !maps_fifo_recv_inplace_ok(plan, i))
            continue;
        inplace[i] = MAPS_FIFO_RECV_INPLACE;
        ++num_aliases;
    }
}

/* Non-zero if `sub` covers its whole slice, densely packed. */
static inline uint32_t maps_subslice_whole_dense(const fifo_tile_plan_t *plan,
                                                 const subslice_desc_t *sub)
{
    const slice_desc_t *slice = get_slice((const tile_plan_t *)plan, sub->slice_id);
    if (slice == NULL || sub->rank != slice->rank || sub->elem_bytes != slice->elem_bytes)
        return 0u;
    uint32_t stride = sub->elem_bytes;
    for (uint32_t d = sub->rank; d-- > 0u;) {
        if (sub->offset[d] != 0u || sub->shape[d] != slice->shape[d] ||
            sub->strides_bytes[d] != stride || slice->strides_bytes[d] != stride)
            return 0u;
        stride *= sub->shape[d];
    }
    return 1u;
}

/* Non-zero if op `index` is an IM2COL that only reinterprets its input: 1x1
 * kernel, unit stride, no padding, channels first, over whole dense NCHW and
 * [C, H*W] slices, so its output bytes equal its input bytes.  Such an op is
 * replaced by aliasing the output slice onto the input for the token, provided
 * only this token's operations read the output and nothing rewrites the input. */
static inline uint32_t maps_fifo_op_is_identity_view(const fifo_tile_plan_t *plan,
                                                     uint32_t index)
{
    const op_desc_t *op = &plan->ops[index];
    if (op->kind != OP_IM2COL || op->num_inputs != 1u || op->num_outputs != 1u)
        return 0u;
    const subslice_desc_t *in = &op->inputs[0];
    const subslice_desc_t *out = &op->outputs[0];
    if (in->rank != 4u || out->rank != 2u || in->shape[0] != 1u ||
        in->elem_bytes != out->elem_bytes || in->slice_id == out->slice_id)
        return 0u;
    if (op->params[0] != 0x00010001u || op->params[1] != 0x00010001u ||
        op->params[3] != 0u || op->params[4] != 0u || op->params[7] == 0u)
        return 0u;
    const uint32_t h = in->shape[2], w = in->shape[3];
    if (maps_operation_low_u16(op->params[5]) != h || maps_operation_high_u16(op->params[5]) != w ||
        maps_operation_low_u16(op->params[6]) != h || maps_operation_high_u16(op->params[6]) != w ||
        out->shape[0] != in->shape[1] || out->shape[1] != h * w)
        return 0u;
    const tile_plan_t *tile = (const tile_plan_t *)plan;
    if (maps_operation_global_offset(tile, in, 0u) != 0u ||
        maps_operation_global_offset(tile, in, 2u) != 0u ||
        maps_operation_global_offset(tile, in, 3u) != 0u ||
        maps_operation_global_offset(tile, out, 1u) != 0u ||
        maps_operation_global_offset(tile, in, 1u) != maps_operation_global_offset(tile, out, 0u))
        return 0u;
    if (!maps_subslice_whole_dense(plan, in) || !maps_subslice_whole_dense(plan, out) ||
        !maps_fifo_slice_inplace_ok(plan, out->slice_id))
        return 0u;
    for (uint32_t i = 0; i < plan->num_recvs; ++i)
        if (plan->recvs[i].dst.slice_id == out->slice_id)
            return 0u;
    for (uint32_t j = index + 1u; j < plan->num_ops; ++j)
        if (maps_fifo_op_uses_slice(&plan->ops[j], in->slice_id))
            for (uint32_t o = 0; o < plan->ops[j].num_outputs; ++o)
                if (plan->ops[j].outputs[o].slice_id == in->slice_id)
                    return 0u;
    return 1u;
}

/* Run op `index`, or alias its output when it is an identity view. */
static inline void maps_fifo_execute_op(const fifo_tile_plan_t *plan, uint32_t index,
                                        uint32_t slot, uint8_t identity_view)
{
    const op_desc_t *op = &plan->ops[index];
    if (identity_view &&
        maps_slice_alias_table()->count < MAPS_MAX_SLICE_ALIASES) {
        maps_slice_alias_push(plan->hartid, op->outputs[0].slice_id, slot,
                              local_subslice_addr((const tile_plan_t *)plan,
                                                  &op->inputs[0], slot));
        return;
    }
    if (maps_execute_operation((const tile_plan_t *)plan, op, slot,
                               plan->operation_runtime) != 0)
        maps_trap();
}

/* Return the FIFO slots held by in-place receives once the token no longer
 * reads them, and restore the planned slice addresses. */
static inline void maps_fifo_release_inplace(const fifo_tile_plan_t *plan,
                                             const uint32_t *held, uint32_t num_held)
{
    for (uint32_t i = 0; i < num_held; ++i)
        fifo_release(plan->hartid, held[i]);
    maps_slice_alias_clear(plan->hartid);
}
#endif

static inline void maps_fifo_init(const fifo_tile_plan_t *plan)
{
    uint32_t data_offset = plan->l1_data_base - get_l1_base(plan->hartid);
    uint32_t fifo_bytes = FIFO_HEADER_SIZE +
        plan->fifo.num_producers * FIFO_RING_STATE_SIZE;
    if (plan->fifo.slot_data_sizes) {
        fifo_bytes += (plan->fifo.num_producers + 1u) * FIFO_SLOT_OFFSET_SIZE;
        for (uint32_t producer = 0u; producer < plan->fifo.num_producers; ++producer)
            fifo_bytes += plan->fifo.num_slots * (FIFO_SLOT_META_SIZE +
                ((plan->fifo.slot_data_sizes[producer] + 3u) & ~3u));
    } else {
        fifo_bytes += plan->fifo.num_producers * plan->fifo.num_slots *
            (FIFO_SLOT_META_SIZE + ((plan->fifo.slot_data_size + 3u) & ~3u));
    }

#if MAPS_FIFO_INPLACE_RECV
    maps_slice_alias_reset();
#endif
    if (plan->fifo.num_producers == 0u)
        return;
    /* l1_fifo.h currently places its header at offset zero. */
    if (plan->fifo.l1_offset_bytes != 0u)
        maps_trap();

#if MAPS_FIFO_INPLACE_RECV
    /* The generator decided which producers gather into shared payload rings
     * and told their senders; the layout must be honoured exactly. */
    if (plan->fifo.num_shared_rings != 0u) {
        const uint32_t num_producers = plan->fifo.num_producers;
        uint32_t sizes[num_producers];
        for (uint32_t p = 0; p < num_producers; ++p)
            sizes[p] = plan->fifo.payload_rings[p] != 0u ? 0u :
                plan->fifo.slot_data_sizes ? plan->fifo.slot_data_sizes[p] :
                plan->fifo.slot_data_size;
        if (fifo_shared_layout_bytes(num_producers, plan->fifo.num_slots, sizes,
                                     plan->fifo.num_shared_rings,
                                     plan->fifo.shared_ring_bytes) > data_offset)
            maps_trap();
        fifo_init_shared(plan->hartid, num_producers, plan->fifo.num_slots, sizes,
                         plan->fifo.payload_rings, plan->fifo.num_shared_rings,
                         plan->fifo.shared_ring_bytes);
        return;
    }
#endif

    if (fifo_bytes > data_offset)
        maps_trap();

    if (plan->fifo.slot_data_sizes)
        fifo_init_variable(plan->hartid, plan->fifo.num_producers,
                           plan->fifo.num_slots, plan->fifo.slot_data_sizes);
    else
        fifo_init(plan->hartid, plan->fifo.num_producers,
                  plan->fifo.num_slots, plan->fifo.slot_data_size);
}

static inline void maps_fifo_check_msg(const fifo_msg_t *msg, const subslice_desc_t *dst)
{
    if (msg->desc.rank != dst->rank || msg->elem_bytes != dst->elem_bytes ||
        msg->desc.num_elems != maps_shape_elems(dst->rank, dst->shape)) {
        maps_trap();
    }
}

/* Copy a packed FIFO payload to a potentially strided MAPS destination. */
static inline void maps_fifo_unpack(const fifo_msg_t *msg, const subslice_desc_t *dst,
                                    uint32_t dst_addr)
{
    maps_fifo_check_msg(msg, dst);

    tensor_sub_slice_t packed;
    fifo_packed_slice(&msg->desc, msg->elem_bytes, &packed);

    tensor_sub_slice_t destination = {
        .rank = dst->rank,
        .num_elems = msg->desc.num_elems,
    };
    for (uint32_t dimension = 0u; dimension < dst->rank; ++dimension) {
        destination.dims[dimension].start = 0u;
        destination.dims[dimension].length = dst->shape[dimension];
        destination.dims[dimension].stride = dst->strides_bytes[dimension];
    }

    if (idma_memcpy_md_to_nd(1u, dst_addr, msg->data_ptr, &packed, &destination,
            msg->elem_bytes) != 0)
        maps_trap();
}

/* Fill the push request for `send`.  The generator decided whether the
 * receiver gathers this piece in place and where it lands, so no remote header
 * or destination-layout reads are needed; copy_dst is the piece's layout in the
 * destination slice. */
static inline void maps_fifo_send_req(const fifo_tile_plan_t *plan,
                                      const fifo_send_desc_t *send,
                                      uint32_t slot,
                                      fifo_push_req_t *req)
{
    *req = (fifo_push_req_t){
        .target_hartid    = send->dst_hartid,
        .producer_idx     = send->producer_idx,
        .src_base_addr    = local_subslice_addr((const tile_plan_t *)plan, &send->src, slot),
        .src              = &send->copy_src,
        .desc             = &send->copy_dst,
        .tag              = maps_fifo_tag(send->transition_id, slot),
        .elem_bytes       = send->src.elem_bytes,
        .dst              = &send->copy_dst,
        .dst_offset_bytes = send->dst_offset_bytes,
        .payload_mode     = send->dst_payload_mode,
        .tail_only        = 1u,
    };
}

static inline void maps_fifo_issue_send(const fifo_tile_plan_t *plan,
                                        const fifo_send_desc_t *send,
                                        uint32_t token,
                                        uint32_t slot)
{
    maps_trace_event((const tile_plan_t *)plan, token, slot, "fifo-send", send->transition_id);
    fifo_push_req_t req;
    maps_fifo_send_req(plan, send, slot, &req);

    if (fifo_push(&req) != 0)
        maps_trap();
    maps_trace_event((const tile_plan_t *)plan, token, slot, "fifo-sent", send->transition_id);
}

static inline int maps_fifo_start_send(const fifo_tile_plan_t *plan,
                                       const fifo_send_desc_t *send,
                                       uint32_t token,
                                       uint32_t slot,
                                       uint32_t reserved,
                                       uint32_t ahead,
                                       maps_fifo_pending_send_t *pending)
{
    fifo_push_req_t req;
    maps_fifo_send_req(plan, send, slot, &req);
    req.ahead = ahead;

    uint32_t start_cycle = maps_read_cycle();
    int rc = fifo_push_async_start_reserved(&req, &pending->push, reserved);
    if (rc != FIFO_ASYNC_STARTED)
        return rc;

    maps_trace_event((const tile_plan_t *)plan, token, slot, "fifo-send",
                     send->transition_id);
    pending->next = 0;
    pending->token = token;
    pending->slot = slot;
    pending->transition_id = send->transition_id;
    pending->start_cycle = start_cycle;
    return FIFO_ASYNC_STARTED;
}

static inline void maps_fifo_unlink_same(maps_fifo_pending_send_t *pending)
{
    if (pending->next_same != 0)
        pending->next_same->prev_same = 0;
    pending->next_same = 0;
    pending->prev_same = 0;
}

static inline void maps_fifo_finish_send(const fifo_tile_plan_t *plan,
                                         maps_fifo_pending_send_t *pending)
{
    if (!pending->push.active)
        return;
    /* Publish older pushes on the same sub-ring first. */
    while (pending->prev_same != 0)
        maps_fifo_finish_send(plan, pending->prev_same);
    if (fifo_push_async_finish(&pending->push) != 0)
        maps_trap();
    maps_fifo_unlink_same(pending);
    maps_trace_event((const tile_plan_t *)plan, pending->token, pending->slot,
                     "fifo-sent", pending->transition_id);
    maps_trace_duration((const tile_plan_t *)plan, pending->token, pending->slot,
                        "send", pending->transition_id,
                        maps_read_cycle() - pending->start_cycle);
}

static inline void maps_fifo_publish_completed_send(
    const fifo_tile_plan_t *plan, maps_fifo_pending_send_t *pending)
{
    if (pending->prev_same != 0)
        return; /* An older push on this sub-ring is still unpublished. */
    if (!fifo_push_async_publish_completed(&pending->push))
        return;
    maps_fifo_unlink_same(pending);
    maps_trace_event((const tile_plan_t *)plan, pending->token, pending->slot,
                     "fifo-sent", pending->transition_id);
    maps_trace_duration((const tile_plan_t *)plan, pending->token, pending->slot,
                        "send", pending->transition_id,
                        maps_read_cycle() - pending->start_cycle);
}

static inline void maps_fifo_reap_completed_sends(
    const fifo_tile_plan_t *plan, maps_fifo_token_state_t *states)
{
    for (uint32_t slot = 0u; slot < plan->num_token_slots; ++slot) {
        maps_fifo_pending_send_t **link = &states[slot].sends;
        while (*link != 0) {
            maps_fifo_pending_send_t *pending = *link;
            if (pending->push.active) {
                if (!fifo_push_async_is_done(&pending->push)) {
                    link = &pending->next;
                    continue;
                }
                maps_fifo_publish_completed_send(plan, pending);
                if (pending->push.active) { /* waits for an older push */
                    link = &pending->next;
                    continue;
                }
            }
            *link = pending->next;
            pending->next = 0;
            --states[slot].num_sends;
        }
    }
}

static inline void maps_fifo_finish_slot_sends(
    const fifo_tile_plan_t *plan,
    maps_fifo_token_state_t *state)
{
    while (state->sends != 0) {
        maps_fifo_pending_send_t *pending = state->sends;
        state->sends = pending->next;
        pending->next = 0;
        --state->num_sends;
        maps_fifo_finish_send(plan, pending);
    }
}

static inline void maps_fifo_finish_all_sends(
    const fifo_tile_plan_t *plan,
    maps_fifo_token_state_t *states)
{
    for (uint32_t slot = 0u; slot < plan->num_token_slots; ++slot)
        maps_fifo_finish_slot_sends(plan, &states[slot]);
}

static inline uint32_t maps_fifo_finish_one_send(
    const fifo_tile_plan_t *plan,
    maps_fifo_token_state_t *states)
{
    for (uint32_t slot = 0u; slot < plan->num_token_slots; ++slot) {
        if (states[slot].sends == 0)
            continue;
        maps_fifo_pending_send_t *pending = states[slot].sends;
        states[slot].sends = pending->next;
        pending->next = 0;
        --states[slot].num_sends;
        maps_fifo_finish_send(plan, pending);
        return 1u;
    }
    return 0u;
}

static inline void maps_fifo_finish_subring_sends(
    const fifo_tile_plan_t *plan,
    maps_fifo_token_state_t *states, uint32_t target_hartid,
    uint32_t producer_idx)
{
    for (uint32_t slot = 0u; slot < plan->num_token_slots; ++slot) {
        maps_fifo_pending_send_t **link = &states[slot].sends;
        while (*link != 0) {
            maps_fifo_pending_send_t *pending = *link;
            if (pending->push.target_hartid != target_hartid ||
                pending->push.producer_idx != producer_idx) {
                link = &pending->next;
                continue;
            }
            *link = pending->next;
            pending->next = 0;
            --states[slot].num_sends;
            maps_fifo_finish_send(plan, pending);
        }
    }
}

/* Youngest unpublished push on a consumer sub-ring (0 if none), and in
 * *count how many pushes on it are unpublished. */
static inline maps_fifo_pending_send_t *maps_fifo_subring_newest(
    maps_fifo_pending_send_t pending[MAPS_FIFO_MAX_PENDING_SENDS],
    uint32_t target_hartid, uint32_t producer_idx, uint32_t *count)
{
    maps_fifo_pending_send_t *newest = 0;
    *count = 0u;
    for (uint32_t index = 0u; index < MAPS_FIFO_MAX_PENDING_SENDS; ++index) {
        maps_fifo_pending_send_t *record = &pending[index];
        if (!record->push.active || record->push.target_hartid != target_hartid ||
            record->push.producer_idx != producer_idx)
            continue;
        ++*count;
        if (record->next_same == 0)
            newest = record;
    }
    return newest;
}

static inline maps_fifo_pending_send_t *maps_fifo_alloc_pending_send(
    maps_fifo_pending_send_t pending[MAPS_FIFO_MAX_PENDING_SENDS])
{
    for (uint32_t index = 0u; index < MAPS_FIFO_MAX_PENDING_SENDS; ++index)
        if (!pending[index].push.active && pending[index].next == 0)
            return &pending[index];
    return 0;
}

/* MAPS receives are dependency-addressed, so inspect the requested producer's
 * sub-ring directly instead of accepting an unrelated ready message from the
 * FIFO's fair round-robin scan. */
static inline uint32_t maps_fifo_peek_from(const fifo_tile_plan_t *plan,
                                           const fifo_recv_desc_t *recv,
                                           fifo_msg_t *out)
{
    const uint32_t producer_idx = recv->producer_idx;
    fifo_header_t *hdr = fifo_get_header(plan->hartid);
    fifo_ring_state_t *rs = fifo_ring_state(hdr, producer_idx);

    if (rs->tail == rs->head)
        return 0u;

    asm volatile("fence r, r" ::: "memory");

    /* Senders publish only the tail: every message on this sub-ring is the
     * receive's piece, laid out as its destination, so the metadata comes from
     * the plan instead of the slot. */
    const subslice_desc_t *dst = &recv->dst;
    out->data_ptr   = (uint32_t)fifo_payload_at(hdr, producer_idx, rs->head);
    out->src        = producer_idx;
    out->elem_bytes = dst->elem_bytes;
    out->desc.rank  = dst->rank;
    out->desc.num_elems = maps_shape_elems(dst->rank, dst->shape);
    for (uint32_t d = 0; d < dst->rank; ++d) {
        out->desc.dims[d].start = 0u;
        out->desc.dims[d].length = dst->shape[d];
        out->desc.dims[d].stride = dst->strides_bytes[d];
    }
    out->data_size = out->desc.num_elems * dst->elem_bytes;
    return 1u;
}

/* `inplace` is the receive's MAPS_FIFO_RECV_* mode.  Returns 1 when the
 * payload was left in FIFO memory (in-place or gathered receive); the caller
 * must then release recv->producer_idx with maps_fifo_release_inplace. */
static inline uint32_t maps_fifo_wait_recv(const fifo_tile_plan_t *plan,
                                           const fifo_recv_desc_t *recv,
                                           uint32_t token,
                                           uint32_t slot,
                                           uint32_t inplace,
                                           maps_fifo_token_state_t *states)
{
    fifo_msg_t msg;

    maps_trace_event((const tile_plan_t *)plan, token, slot, "fifo-wait",
                     recv->transition_id);

    if (states != 0)
        maps_fifo_reap_completed_sends(plan, states);
    for (;;) {
        if (!maps_fifo_peek_from(plan, recv, &msg)) {
            /* Keep completed outgoing transfers moving while this tile waits
             * for an input.  Blocking on every outstanding send here defeats
             * the overlap provided by asynchronous FIFO pushes. */
            if (states != 0)
                maps_fifo_reap_completed_sends(plan, states);
            continue;
        }

#if MAPS_FIFO_INPLACE_RECV
        if (inplace == MAPS_FIFO_RECV_GATHER) {
            /* The piece already sits at its place in the shared slot, which
             * every receive of the slice must agree on. */
            uint32_t addr;
            maps_fifo_check_msg(&msg, &recv->dst);
            if (maps_slice_alias_lookup(plan->hartid, recv->dst.slice_id, slot, &addr)) {
                if (addr != msg.data_ptr)
                    maps_trap();
            } else {
                maps_slice_alias_push(plan->hartid, recv->dst.slice_id, slot, msg.data_ptr);
            }
            maps_trace_event((const tile_plan_t *)plan, token, slot, "fifo-recv-gather",
                             recv->transition_id);
            return 1u;
        }
        /* Spatz vector accesses must stay 4-byte aligned; FIFO payloads are
         * by construction, so the unpack fallback should never trigger. */
        if (inplace == MAPS_FIFO_RECV_INPLACE && (msg.data_ptr & 3u) == 0u) {
            maps_fifo_check_msg(&msg, &recv->dst);
            maps_slice_alias_push(plan->hartid, recv->dst.slice_id, slot, msg.data_ptr);
            maps_trace_event((const tile_plan_t *)plan, token, slot, "fifo-recv-inplace",
                             recv->transition_id);
            return 1u;
        }
#else
        (void)inplace;
#endif

        maps_fifo_unpack(
            &msg, &recv->dst,
            local_subslice_addr((const tile_plan_t *)plan, &recv->dst, slot));
        fifo_release(plan->hartid, msg.src);
        maps_trace_event((const tile_plan_t *)plan, token, slot, "fifo-recv",
                         recv->transition_id);
        return 0u;
    }
}

/* Copy one descriptor array to *next in L1 and return its new address. */
static inline const void *maps_fifo_array_to_l1(const void *src, uint32_t bytes,
                                                uint32_t *next)
{
    if (src == 0 || bytes == 0u)
        return src;
    const uint32_t dst = *next;
    /* dir=0: AXI (L2) -> OBI (local L1). */
    if (maps_idma_copy_1d_ticketed(0u, (uint32_t)src, dst, bytes) != 0)
        maps_trap();
    *next = (dst + bytes + 15u) & ~15u;
    return (const void *)dst;
}

/* The generated descriptors are `static const` data in L2, so every field the
 * control core reads during a token is a load over the NoC.  Copy them into
 * this tile's L1 once and repoint the plan.  The generator reserves the region
 * after the planned slice data and checks it fits below the task scratch. */
static inline void maps_fifo_plan_to_l1(fifo_tile_plan_t *plan)
{
    if (plan->descriptor_l1_offset == 0u)
        return;
    uint32_t next = get_l1_base(plan->hartid) + plan->descriptor_l1_offset;
#define MAPS_FIFO_ARRAY_TO_L1(field, count)                                        \
    plan->field = maps_fifo_array_to_l1(plan->field, (count) * sizeof(*plan->field), \
                                        &next)
    MAPS_FIFO_ARRAY_TO_L1(slices, plan->num_slices);
    MAPS_FIFO_ARRAY_TO_L1(init_l2_reads, plan->num_init_l2_reads);
    MAPS_FIFO_ARRAY_TO_L1(l2_reads, plan->num_l2_reads);
    MAPS_FIFO_ARRAY_TO_L1(recvs, plan->num_recvs);
    MAPS_FIFO_ARRAY_TO_L1(ops, plan->num_ops);
    MAPS_FIFO_ARRAY_TO_L1(sends, plan->num_sends);
    MAPS_FIFO_ARRAY_TO_L1(l2_writes, plan->num_l2_writes);
    MAPS_FIFO_ARRAY_TO_L1(fifo.slot_data_sizes, plan->fifo.num_producers);
#undef MAPS_FIFO_ARRAY_TO_L1
}

static inline void maps_fifo_init_tile(const fifo_tile_plan_t *plan)
{
    for (uint32_t i = 0; i < plan->num_init_l2_reads; ++i)
        issue_l2_read_token((const tile_plan_t *)plan, &plan->init_l2_reads[i], 0u, 0u);
}

static inline void maps_fifo_execute_operation(const fifo_tile_plan_t *plan,
                                               uint32_t token, uint32_t slot,
                                               uint32_t index, uint8_t identity_view)
{
    const op_desc_t *op = &plan->ops[index];
#if MAPS_KERNEL_PROGRESS
    const uint32_t start = maps_read_cycle();
    printf("MAPS_KERNEL_BEGIN tile=%u token=%u slot=%u op=%u kind=%u cycle=%u\n",
           plan->hartid, token, slot, index, op->kind, start);
#endif
    int result = 0;
#if MAPS_FIFO_INPLACE_RECV
    if (identity_view && maps_slice_alias_table()->count < MAPS_MAX_SLICE_ALIASES) {
        maps_slice_alias_push(plan->hartid, op->outputs[0].slice_id, slot,
                              local_subslice_addr((const tile_plan_t *)plan,
                                                  &op->inputs[0], slot));
    } else
#endif
    {
        result = maps_execute_operation((const tile_plan_t *)plan, op, slot,
                                        plan->operation_runtime);
    }
    if (result != 0) {
        printf("MAPS_KERNEL_ERROR tile=%u token=%u slot=%u op=%u kind=%u code=%d cycle=%u\n",
               plan->hartid, token, slot, index, op->kind, result, maps_read_cycle());
        maps_trap();
    }
#if MAPS_KERNEL_PROGRESS
    printf("MAPS_KERNEL_DONE tile=%u token=%u slot=%u op=%u kind=%u cycles=%u cycle=%u\n",
           plan->hartid, token, slot, index, op->kind, maps_read_cycle() - start, maps_read_cycle());
#endif
}

static inline void maps_fifo_run_tile_token(const fifo_tile_plan_t *plan, uint32_t token)
{
    uint32_t slot = maps_token_slot((const tile_plan_t *)plan, token);
    uint32_t token_start = maps_read_cycle();

    for (uint32_t i = 0; i < plan->num_l2_reads; ++i) {
        uint32_t step_start = maps_read_cycle();
        issue_l2_read_token((const tile_plan_t *)plan, &plan->l2_reads[i], token, slot);
        maps_trace_duration((const tile_plan_t *)plan, token, slot, "l2-read", i,
                            maps_read_cycle() - step_start);
    }
#if MAPS_FIFO_INPLACE_RECV
    uint8_t inplace[plan->num_recvs + 1u];
    uint32_t held[plan->num_recvs + 1u];
    uint32_t num_held = 0u;
    maps_fifo_plan_inplace(plan, inplace);
#endif
    for (uint32_t i = 0; i < plan->num_recvs; ++i) {
        uint32_t step_start = maps_read_cycle();
#if MAPS_FIFO_INPLACE_RECV
        if (maps_fifo_wait_recv(plan, &plan->recvs[i], token, slot, inplace[i], 0))
            held[num_held++] = plan->recvs[i].producer_idx;
#else
        maps_fifo_wait_recv(
            plan, &plan->recvs[i], token, slot, 0u, 0);
#endif
        maps_trace_duration((const tile_plan_t *)plan, token, slot, "recv",
                            plan->recvs[i].transition_id,
                            maps_read_cycle() - step_start);
    }
    for (uint32_t i = 0; i < plan->num_ops; ++i) {
        uint32_t step_start = maps_read_cycle();
        maps_fifo_execute_operation(plan, token, slot, i,
#if MAPS_FIFO_INPLACE_RECV
            (uint8_t)maps_fifo_op_is_identity_view(plan, i)
#else
            0u
#endif
        );
        maps_trace_duration((const tile_plan_t *)plan, token, slot, "op", i,
                            maps_read_cycle() - step_start);
    }
#if MAPS_FIFO_INPLACE_RECV
    maps_fifo_release_inplace(plan, held, num_held);
#endif
    for (uint32_t i = 0; i < plan->num_sends; ++i) {
        uint32_t step_start = maps_read_cycle();
        maps_fifo_issue_send(plan, &plan->sends[i], token, slot);
        maps_trace_duration((const tile_plan_t *)plan, token, slot, "send",
                            plan->sends[i].transition_id,
                            maps_read_cycle() - step_start);
    }
    for (uint32_t i = 0; i < plan->num_l2_writes; ++i) {
        uint32_t step_start = maps_read_cycle();
        issue_l2_write_token((const tile_plan_t *)plan, &plan->l2_writes[i], token, slot);
        maps_trace_duration((const tile_plan_t *)plan, token, slot, "l2-write", i,
                            maps_read_cycle() - step_start);
    }

    uint32_t token_end = maps_read_cycle();
    maps_trace_duration((const tile_plan_t *)plan, token, slot, "token", token,
                        token_end - token_start);
#if MAPS_TOKEN_PROGRESS
    printf("MAPS_TOKEN_DONE tile=%u token=%u cycle=%u\n",
           plan->hartid, token, token_end);
#endif
}

/* While a Spatz task runs, the control core reaps finished sends and reserves
 * the token's outgoing FIFO slots, so only the DMA submissions are left once
 * the operations are done. */
typedef struct {
    const fifo_tile_plan_t *plan;
    maps_fifo_token_state_t *states;
    const maps_fifo_pending_send_t *pending;
    uint32_t *reserved; /* Per send: reserved payload address, 0 if none yet. */
    uint32_t next;      /* Next send to try to reserve. */
    /* Pre-received inputs of the next token: for receives held in place, the
     * payload address of the next message once it is published and its tag
     * checked (0 if not yet); the next token then only takes the alias. */
    const uint8_t *inplace;
    uint32_t *next_payload;
    uint32_t next_slot;
    uint32_t next_token_exists;
} maps_fifo_background_t;

static inline uint32_t maps_fifo_subring_busy(const maps_fifo_pending_send_t *pending,
                                              uint32_t target_hartid, uint32_t producer_idx)
{
    for (uint32_t i = 0u; i < MAPS_FIFO_MAX_PENDING_SENDS; ++i)
        if (pending[i].push.active && pending[i].push.target_hartid == target_hartid &&
            pending[i].push.producer_idx == producer_idx)
            return 1u;
    return 0u;
}

static inline uint32_t maps_fifo_background(void *context)
{
    maps_fifo_background_t *bg = (maps_fifo_background_t *)context;
    const fifo_tile_plan_t *plan = bg->plan;
    maps_fifo_reap_completed_sends(plan, bg->states);
    while (bg->next < plan->num_sends) {
        const uint32_t i = bg->next++;
        const fifo_send_desc_t *send = &plan->sends[i];
        /* An unpublished push on the sub-ring has not advanced its tail yet,
         * so the slot cannot be known before that push is published. */
        if (bg->reserved[i] != 0u ||
            maps_fifo_subring_busy(bg->pending, send->dst_hartid, send->producer_idx))
            continue;
        void *payload = 0;
        if (fifo_try_reserve_mode(send->dst_hartid, send->producer_idx,
                                  send->dst_payload_mode, &payload)) {
            bg->reserved[i] = (uint32_t)payload;
            return 1u;
        }
    }
#if MAPS_FIFO_INPLACE_RECV
    /* The current token still holds its in-place message at the head of each
     * such sub-ring; the next token's message is the one after it. */
    if (!bg->next_token_exists)
        return 0u;
    uint32_t pending_inputs = 0u;
    fifo_header_t *hdr = fifo_get_header(plan->hartid);
    for (uint32_t i = 0; i < plan->num_recvs; ++i) {
        if (bg->inplace[i] != MAPS_FIFO_RECV_INPLACE || bg->next_payload[i] != 0u)
            continue;
        const fifo_recv_desc_t *recv = &plan->recvs[i];
        const fifo_ring_state_t *rs = fifo_ring_state(hdr, recv->producer_idx);
        const uint32_t counter = rs->head + 1u;
        if ((int32_t)(rs->tail - counter) <= 0) {
            pending_inputs = 1u;
            continue;
        }
        /* Senders publish only the tail (no slot metadata), so the message is
         * identified by its sub-ring position alone, as in maps_fifo_peek_from. */
        asm volatile("fence r, r" ::: "memory");
        const uint32_t payload =
            (uint32_t)fifo_payload_at(hdr, recv->producer_idx, counter);
        if ((payload & 3u) != 0u)
            continue; /* Left to the normal receive path. */
        bg->next_payload[i] = payload;
        return 1u;
    }
    return pending_inputs;
#else
    return 0u;
#endif
}

/* Non-zero if L2 read `index` may be issued a token ahead: nothing but the
 * read writes its destination slice, and no send reads it (an asynchronous
 * send of an earlier token could still be reading the slot it overwrites). */
static inline uint32_t maps_fifo_l2_read_prefetchable(const fifo_tile_plan_t *plan,
                                                      uint32_t index)
{
    const uint32_t slice_id = plan->l2_reads[index].dst.slice_id;
    for (uint32_t i = 0; i < plan->num_sends; ++i)
        if (plan->sends[i].src.slice_id == slice_id)
            return 0u;
    for (uint32_t i = 0; i < plan->num_recvs; ++i)
        if (plan->recvs[i].dst.slice_id == slice_id)
            return 0u;
    for (uint32_t i = 0; i < plan->num_ops; ++i)
        for (uint32_t o = 0; o < plan->ops[i].num_outputs; ++o)
            if (plan->ops[i].outputs[o].slice_id == slice_id)
                return 0u;
    return 1u;
}

/* Submit L2 read `index` of `token` without waiting; 0 if it must be issued
 * blocking instead (layout not one descriptor, or the queue is full). */
static inline idma_ticket_t maps_fifo_prefetch_l2_read(const fifo_tile_plan_t *plan,
                                                       uint32_t index, uint32_t token)
{
    const l2_read_desc_t *read = &plan->l2_reads[index];
    const uint32_t slot = maps_token_slot((const tile_plan_t *)plan, token);
    const idma_ticket_t ticket = maps_idma_submit_md_to_nd(0u, local_subslice_addr((const tile_plan_t *)plan, &read->dst, slot),
        read->src_l2_addr + token * read->src_l2_token_stride_bytes,
        &read->copy_src, &read->copy_dst, read->dst.elem_bytes);
    return ticket == MAPS_IDMA_UNSUPPORTED_TICKET ? 0u : ticket;
}

static inline void maps_fifo_run_tile_tokens(const fifo_tile_plan_t *plan, uint32_t num_tokens)
{
    uint32_t run_start = maps_read_cycle();
#ifdef MAPS_EXPERIMENT_TRACE
    if (num_tokens > MAPS_EXPERIMENT_MAX_TOKENS)
        maps_trap();
    maps_fifo_experiment_trace_t *trace = maps_fifo_experiment_trace();
#endif

    if (plan->num_token_slots == 0u)
        maps_trap();

    /* Only the per-slot list heads scale with the execution plan.  Transfer
     * records are bounded by the hardware's 16-entry descriptor queue. */
    maps_fifo_token_state_t states[plan->num_token_slots];
    maps_fifo_pending_send_t pending[MAPS_FIFO_MAX_PENDING_SENDS] = {0};
    uint32_t reserved[plan->num_sends + 1u];
    /* L2 reads of the next token, submitted while this token's operations run:
     * the DMA ticket, or 0 when the read is issued blocking as usual. */
    uint8_t l2_prefetchable[plan->num_l2_reads + 1u];
    idma_ticket_t l2_prefetched[plan->num_l2_reads + 1u];
    for (uint32_t i = 0; i < plan->num_l2_reads; ++i) {
        l2_prefetchable[i] = (uint8_t)maps_fifo_l2_read_prefetchable(plan, i);
        l2_prefetched[i] = 0u;
    }
    uint32_t next_payload[plan->num_recvs + 1u];
    uint32_t this_payload[plan->num_recvs + 1u];
    /* Every field initialised explicitly: CORE-V GCC 14.1 ICEs (maybe_postinc,
     * with xcvmem) when zero-filling the remaining fields of this initialiser. */
    maps_fifo_background_t background = {
        .plan = plan, .states = states, .pending = pending, .reserved = reserved,
        .next = 0u, .inplace = 0, .next_payload = next_payload, .next_slot = 0u,
        .next_token_exists = 0u,
    };
    for (uint32_t i = 0; i < plan->num_recvs; ++i)
        next_payload[i] = 0u;
#if MAPS_FIFO_INPLACE_RECV
    uint8_t inplace[plan->num_recvs + 1u];
    uint32_t held[plan->num_recvs + 1u];
    uint8_t identity_view[plan->num_ops + 1u];
    maps_fifo_plan_inplace(plan, inplace);
    for (uint32_t i = 0; i < plan->num_ops; ++i)
        identity_view[i] = (uint8_t)maps_fifo_op_is_identity_view(plan, i);
    background.inplace = inplace;
#endif
    for (uint32_t slot = 0u; slot < plan->num_token_slots; ++slot) {
        states[slot].sends = 0;
        states[slot].num_sends = 0u;
    }

    for (uint32_t token = 0; token < num_tokens; ++token) {
        uint32_t slot = maps_token_slot((const tile_plan_t *)plan, token);
#ifdef MAPS_EXPERIMENT_TRACE
        trace->starts[plan->hartid][token] = maps_read_cycle();
#endif

        /* Token N reuses token N-B's backing storage.  Every DMA still reading
         * that slot must be complete and published before the first overwrite. */
        maps_fifo_finish_slot_sends(plan, &states[slot]);

        for (uint32_t i = 0; i < plan->num_l2_reads; ++i) {
            uint32_t step_start = maps_read_cycle();
            if (l2_prefetched[i] != 0u)
                maps_idma_wait_ticket(0u, l2_prefetched[i]);
            else
                issue_l2_read_token((const tile_plan_t *)plan, &plan->l2_reads[i], token,
                                    slot);
            l2_prefetched[i] = 0u;
            maps_trace_duration((const tile_plan_t *)plan, token, slot, "l2-read", i,
                                maps_read_cycle() - step_start);
        }

        maps_fifo_reap_completed_sends(plan, states);
#if MAPS_FIFO_INPLACE_RECV
        uint32_t num_held = 0u;
        for (uint32_t i = 0; i < plan->num_recvs; ++i) {
            this_payload[i] = next_payload[i];
            next_payload[i] = 0u;
        }
#endif
        for (uint32_t i = 0; i < plan->num_recvs; ++i) {
            uint32_t step_start = maps_read_cycle();
#if MAPS_FIFO_INPLACE_RECV
            if (this_payload[i] != 0u) {
                /* Pre-received during the previous token's operations: the
                 * message is now at the head of its sub-ring. */
                maps_slice_alias_push(plan->hartid, plan->recvs[i].dst.slice_id, slot,
                                      this_payload[i]);
                held[num_held++] = plan->recvs[i].producer_idx;
            } else if (maps_fifo_wait_recv(plan, &plan->recvs[i], token, slot, inplace[i], states))
                held[num_held++] = plan->recvs[i].producer_idx;
#else
            maps_fifo_wait_recv(plan, &plan->recvs[i], token, slot, 0u, states);
#endif
            maps_trace_duration((const tile_plan_t *)plan, token, slot, "recv",
                                plan->recvs[i].transition_id,
                                maps_read_cycle() - step_start);
        }
        maps_fifo_reap_completed_sends(plan, states);
        for (uint32_t i = 0; i < plan->num_sends; ++i)
            reserved[i] = 0u;
        /* The next token's L2 inputs go to the other token slot, which no one
         * uses during this token, so they can stream in under the operations. */
        if (token + 1u < num_tokens && plan->num_token_slots > 1u)
            for (uint32_t i = 0; i < plan->num_l2_reads; ++i)
                if (l2_prefetchable[i])
                    l2_prefetched[i] = maps_fifo_prefetch_l2_read(plan, i, token + 1u);
        background.next = 0u;
        background.next_slot = maps_token_slot((const tile_plan_t *)plan, token + 1u);
        background.next_token_exists = token + 1u < num_tokens;
        if (plan->operation_runtime != 0) {
            plan->operation_runtime->background = maps_fifo_background;
            plan->operation_runtime->background_ctx = &background;
        }
        for (uint32_t i = 0; i < plan->num_ops; ++i) {
            uint32_t step_start = maps_read_cycle();
            maps_fifo_execute_operation(plan, token, slot, i,
#if MAPS_FIFO_INPLACE_RECV
            identity_view[i]
#else
            0u
#endif
        );
            maps_trace_duration((const tile_plan_t *)plan, token, slot, "op", i,
                                maps_read_cycle() - step_start);
        }
        if (plan->operation_runtime != 0)
            plan->operation_runtime->background = 0;
#if MAPS_FIFO_INPLACE_RECV
        maps_fifo_release_inplace(plan, held, num_held);
#endif
        maps_fifo_reap_completed_sends(plan, states);
        for (uint32_t i = 0; i < plan->num_l2_writes; ++i) {
            uint32_t step_start = maps_read_cycle();
            issue_l2_write_token((const tile_plan_t *)plan, &plan->l2_writes[i], token, slot);
            maps_trace_duration((const tile_plan_t *)plan, token, slot, "l2-write", i,
                                maps_read_cycle() - step_start);
        }

        maps_fifo_reap_completed_sends(plan, states);
        for (uint32_t i = 0; i < plan->num_sends; ++i) {
            const fifo_send_desc_t *send = &plan->sends[i];

            /* tail only advances at publication, so a reservation behind
             * unpublished pushes on the same sub-ring takes tail + their
             * count and is published after them (maps_fifo_publish_completed_send). */
            for (;;) {
                maps_fifo_reap_completed_sends(plan, states);
                maps_fifo_pending_send_t *record =
                    maps_fifo_alloc_pending_send(pending);
                if (record == 0) {
                    if (!maps_fifo_finish_one_send(plan, states))
                        maps_trap();
                    continue;
                }

                uint32_t ahead;
                maps_fifo_pending_send_t *older = maps_fifo_subring_newest(
                    pending, send->dst_hartid, send->producer_idx, &ahead);
                int rc = maps_fifo_start_send(
                    plan, send, token, slot, ahead == 0u ? reserved[i] : 0u, ahead, record);
                if (rc == FIFO_ASYNC_STARTED) {
                    record->prev_same = older;
                    record->next_same = 0;
                    if (older != 0)
                        older->next_same = record;
                    maps_fifo_pending_send_t **tail = &states[slot].sends;
                    while (*tail != 0)
                        tail = &(*tail)->next;
                    record->next = 0;
                    *tail = record;
                    ++states[slot].num_sends;
                    break;
                }
                if (rc == FIFO_ASYNC_UNSUPPORTED) {
                    /* The generalized asynchronous mover accepts only layouts
                     * representable by one descriptor.  Drain first, then use
                     * the established blocking rank-N mover. */
                    maps_fifo_finish_all_sends(plan, states);
                    uint32_t step_start = maps_read_cycle();
                    maps_fifo_issue_send(
                        plan, send, token, slot);
                    maps_trace_duration((const tile_plan_t *)plan, token, slot,
                                        "send", send->transition_id,
                                        maps_read_cycle() - step_start);
                    break;
                }
                if (rc == FIFO_ASYNC_IDMA_FULL) {
                    maps_fifo_reap_completed_sends(plan, states);
                    if (!maps_fifo_finish_one_send(plan, states))
                        maps_trap();
                    continue;
                }
                if (rc == FIFO_ASYNC_FIFO_FULL) {
                    maps_fifo_finish_all_sends(plan, states);
                    while (fifo_producer_is_full(
                               send->dst_hartid, send->producer_idx))
                        __asm__ volatile("" ::: "memory");
                    continue;
                }
                maps_trap();
            }
        }
        maps_fifo_reap_completed_sends(plan, states);
#ifdef MAPS_EXPERIMENT_TRACE
        trace->ends[plan->hartid][token] = maps_read_cycle();
#endif
#if MAPS_TOKEN_PROGRESS
        /* Processing is complete; asynchronous outgoing transfers may still be in flight. */
        printf("MAPS_TOKEN_DONE tile=%u token=%u cycle=%u\n",
               plan->hartid, token, maps_read_cycle());
#endif
    }
    maps_fifo_finish_all_sends(plan, states);

    maps_trace_duration((const tile_plan_t *)plan, 0u, 0u, "run", num_tokens,
                        maps_read_cycle() - run_start);
}

static inline void maps_fifo_flush_experiment_trace(const fifo_tile_plan_t *plan,
                                                    uint32_t num_tokens)
{
#ifdef MAPS_EXPERIMENT_TRACE
    maps_fifo_experiment_trace_t *trace = maps_fifo_experiment_trace();
    maps_experiment_duration_trace_t *durations = maps_experiment_duration_trace();
    for (uint32_t token = 0; token < num_tokens; ++token)
        printf("MAPS_TOKEN tile=%u token=%u start=%u end=%u output=%u\n",
               plan->hartid, token, trace->starts[plan->hartid][token],
               trace->ends[plan->hartid][token], plan->num_l2_writes != 0u);
    uint32_t count = durations->counts[plan->hartid];
    if (count > MAPS_EXPERIMENT_MAX_DURATION_EVENTS)
        count = MAPS_EXPERIMENT_MAX_DURATION_EVENTS;
    for (uint32_t event = 0u; event < count; ++event) {
        const maps_experiment_duration_event_t *entry =
            &durations->events[plan->hartid][event];
        static const char *const phase_names[] = {
            "op", "send", "recv", "l2-read", "l2-write", "token", "run"
        };
        const char *phase = entry->phase < 7u ? phase_names[entry->phase] : "unknown";
        printf("maps t%u tok %u slot %u %s %u start %u end %u cycles %u\n", plan->hartid,
               entry->token, entry->slot, phase, entry->index, entry->start_cycle,
               entry->end_cycle, entry->cycles);
    }
#else
    (void)plan;
    (void)num_tokens;
#endif
}

#endif /* MAPS_UTILS_V2_H */
