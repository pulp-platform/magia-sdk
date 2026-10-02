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
 */

#include "utils/maps_utils.h"
#include "utils/maps_operations.h"
#include "utils/l1_fifo.h"

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
} fifo_send_desc_t;

typedef struct {
    uint32_t l1_offset_bytes;
    uint32_t num_producers;
    uint32_t num_slots;
    const uint32_t *slot_data_sizes;
    uint32_t slot_data_size;
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
} fifo_tile_plan_t;

typedef struct maps_fifo_pending_send {
    fifo_pending_push_t push;
    struct maps_fifo_pending_send *next;
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

    if (plan->fifo.num_producers == 0u)
        return;
    /* l1_fifo.h currently places its header at offset zero. */
    if (plan->fifo.l1_offset_bytes != 0u || fifo_bytes > data_offset)
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
                                    uint32_t dst_addr, idma_controller_t *idma_ctrl,
                                    eu_controller_t *eu_ctrl)
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

    if (idma_memcpy_md_to_nd(
            idma_ctrl, 1u, dst_addr, msg->data_ptr, &packed, &destination,
            msg->elem_bytes, eu_ctrl) != 0)
        maps_trap();
}

static inline void maps_fifo_issue_send(const fifo_tile_plan_t *plan,
                                        const fifo_send_desc_t *send,
                                        uint32_t token,
                                        uint32_t slot,
                                        idma_controller_t *idma_ctrl,
                                        eu_controller_t *eu_ctrl)
{
    maps_trace_event((const tile_plan_t *)plan, token, slot, "fifo-send", send->transition_id);
    fifo_push_req_t req = {
        .target_hartid = send->dst_hartid,
        .producer_idx  = send->producer_idx,
        .src_base_addr = local_subslice_addr((const tile_plan_t *)plan, &send->src, slot),
        .src           = &send->copy_src,
        .desc          = &send->copy_dst,
        .tag           = maps_fifo_tag(send->transition_id, slot),
        .elem_bytes    = send->src.elem_bytes,
    };

    if (fifo_push(idma_ctrl, eu_ctrl, &req) != 0)
        maps_trap();
    maps_trace_event((const tile_plan_t *)plan, token, slot, "fifo-sent", send->transition_id);
}

static inline int maps_fifo_start_send(const fifo_tile_plan_t *plan,
                                       const fifo_send_desc_t *send,
                                       uint32_t token,
                                       uint32_t slot,
                                       idma_controller_t *idma_ctrl,
                                       eu_controller_t *eu_ctrl,
                                       maps_fifo_pending_send_t *pending)
{
    fifo_push_req_t req = {
        .target_hartid = send->dst_hartid,
        .producer_idx  = send->producer_idx,
        .src_base_addr = local_subslice_addr((const tile_plan_t *)plan, &send->src, slot),
        .src           = &send->copy_src,
        .desc          = &send->copy_dst,
        .tag           = maps_fifo_tag(send->transition_id, slot),
        .elem_bytes    = send->src.elem_bytes,
    };

    uint32_t start_cycle = maps_read_cycle();
    int rc = fifo_push_async_start(idma_ctrl, &req, &pending->push);
    (void)eu_ctrl;
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

static inline void maps_fifo_finish_send(const fifo_tile_plan_t *plan,
                                         eu_controller_t *eu_ctrl,
                                         maps_fifo_pending_send_t *pending)
{
    if (!pending->push.active)
        return;
    if (fifo_push_async_finish(eu_ctrl, &pending->push) != 0)
        maps_trap();
    maps_trace_event((const tile_plan_t *)plan, pending->token, pending->slot,
                     "fifo-sent", pending->transition_id);
    maps_trace_duration((const tile_plan_t *)plan, pending->token, pending->slot,
                        "send", pending->transition_id,
                        maps_read_cycle() - pending->start_cycle);
}

static inline void maps_fifo_publish_completed_send(
    const fifo_tile_plan_t *plan, maps_fifo_pending_send_t *pending)
{
    if (!fifo_push_async_publish_completed(&pending->push))
        return;
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
            if (!fifo_push_async_is_done(&pending->push)) {
                link = &pending->next;
                continue;
            }
            maps_fifo_publish_completed_send(plan, pending);
            *link = pending->next;
            pending->next = 0;
            --states[slot].num_sends;
        }
    }
}

static inline void maps_fifo_finish_slot_sends(
    const fifo_tile_plan_t *plan, eu_controller_t *eu_ctrl,
    maps_fifo_token_state_t *state)
{
    while (state->sends != 0) {
        maps_fifo_pending_send_t *pending = state->sends;
        state->sends = pending->next;
        pending->next = 0;
        --state->num_sends;
        maps_fifo_finish_send(plan, eu_ctrl, pending);
    }
}

static inline void maps_fifo_finish_all_sends(
    const fifo_tile_plan_t *plan, eu_controller_t *eu_ctrl,
    maps_fifo_token_state_t *states)
{
    for (uint32_t slot = 0u; slot < plan->num_token_slots; ++slot)
        maps_fifo_finish_slot_sends(plan, eu_ctrl, &states[slot]);
}

static inline uint32_t maps_fifo_finish_one_send(
    const fifo_tile_plan_t *plan, eu_controller_t *eu_ctrl,
    maps_fifo_token_state_t *states)
{
    for (uint32_t slot = 0u; slot < plan->num_token_slots; ++slot) {
        if (states[slot].sends == 0)
            continue;
        maps_fifo_pending_send_t *pending = states[slot].sends;
        states[slot].sends = pending->next;
        pending->next = 0;
        --states[slot].num_sends;
        maps_fifo_finish_send(plan, eu_ctrl, pending);
        return 1u;
    }
    return 0u;
}

static inline void maps_fifo_finish_subring_sends(
    const fifo_tile_plan_t *plan, eu_controller_t *eu_ctrl,
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
            maps_fifo_finish_send(plan, eu_ctrl, pending);
        }
    }
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
                                           uint32_t producer_idx,
                                           fifo_msg_t *out)
{
    fifo_header_t *hdr = fifo_get_header(plan->hartid);
    fifo_ring_state_t *rs = fifo_ring_state(hdr, producer_idx);

    if (rs->tail == rs->head)
        return 0u;

    asm volatile("fence r, r" ::: "memory");

    fifo_slot_t *slot = fifo_slot_at(hdr, producer_idx, rs->head);
    out->data_ptr     = (uint32_t)fifo_slot_data(slot);
    out->src          = producer_idx;
    out->tag          = slot->tag;
    out->elem_bytes   = slot->elem_bytes;
    out->data_size    = slot->data_size;
    out->desc         = slot->desc;
    return 1u;
}

#if MAPS_FIFO_INPLACE_RECV
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

/* A receive may be consumed straight from its FIFO slot when its packed
 * payload is byte-identical to the destination slice and only the token's
 * operations read the slice.  The slot is then released right after the
 * operations, before any transfer that could block on backpressure. */
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

    /* A second writer would need the planned storage; a second receive on
     * the same sub-ring would be hidden behind the held head slot. */
    for (uint32_t i = 0; i < plan->num_recvs; ++i) {
        if (i == recv_idx)
            continue;
        if (plan->recvs[i].dst.slice_id == dst->slice_id ||
            plan->recvs[i].producer_idx == recv->producer_idx)
            return 0u;
    }
    for (uint32_t i = 0; i < plan->num_l2_reads; ++i)
        if (plan->l2_reads[i].dst.slice_id == dst->slice_id)
            return 0u;

    /* Transfers out of the slice run after the release (and asynchronous
     * sends outlive the token); collective peers address this tile's slices
     * from the plan rather than through the alias. */
    for (uint32_t i = 0; i < plan->num_sends; ++i)
        if (plan->sends[i].src.slice_id == dst->slice_id)
            return 0u;
    for (uint32_t i = 0; i < plan->num_l2_writes; ++i)
        if (plan->l2_writes[i].src.slice_id == dst->slice_id)
            return 0u;
    for (uint32_t i = 0; i < plan->num_ops; ++i) {
        const op_desc_t *op = &plan->ops[i];
        if ((op->kind == OP_ALL_REDUCE_MAX || op->kind == OP_ALL_REDUCE_SUM ||
             op->collective.num_participants != 0u) &&
            maps_fifo_op_uses_slice(op, dst->slice_id))
            return 0u;
    }
    return 1u;
}

static inline void maps_fifo_plan_inplace(const fifo_tile_plan_t *plan, uint8_t *inplace)
{
    uint32_t num_inplace = 0u;

    for (uint32_t i = 0; i < plan->num_recvs; ++i) {
        inplace[i] = num_inplace < MAPS_MAX_SLICE_ALIASES &&
                     maps_fifo_recv_inplace_ok(plan, i);
        num_inplace += inplace[i];
    }
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

/* Returns 1 when the payload was left in its FIFO slot (in-place receive); the
 * caller must then release recv->producer_idx with maps_fifo_release_inplace. */
static inline uint32_t maps_fifo_wait_recv(const fifo_tile_plan_t *plan,
                                           const fifo_recv_desc_t *recv,
                                           uint32_t token,
                                           uint32_t slot,
                                           uint32_t inplace,
                                           idma_controller_t *idma_ctrl,
                                           eu_controller_t *eu_ctrl,
                                           maps_fifo_token_state_t *states)
{
    fifo_msg_t msg;
    uint32_t expected_tag = maps_fifo_tag(recv->transition_id, slot);

    maps_trace_event((const tile_plan_t *)plan, token, slot, "fifo-wait",
                     recv->transition_id);

    if (states != 0)
        maps_fifo_reap_completed_sends(plan, states);
    for (;;) {
        if (!maps_fifo_peek_from(plan, recv->producer_idx, &msg)) {
            /* Keep completed outgoing transfers moving while this tile waits
             * for an input.  Blocking on every outstanding send here defeats
             * the overlap provided by asynchronous FIFO pushes. */
            if (states != 0)
                maps_fifo_reap_completed_sends(plan, states);
            continue;
        }

        if (msg.tag != expected_tag) {
            maps_trace_event((const tile_plan_t *)plan, token, slot, "fifo-bad-tag",
                             msg.tag);
            maps_trap();
        }

#if MAPS_FIFO_INPLACE_RECV
        /* Spatz vector accesses must stay 4-byte aligned; FIFO payloads are
         * by construction, so the unpack fallback should never trigger. */
        if (inplace && (msg.data_ptr & 3u) == 0u) {
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
            local_subslice_addr((const tile_plan_t *)plan, &recv->dst, slot),
            idma_ctrl, eu_ctrl);
        fifo_release(plan->hartid, msg.src);
        maps_trace_event((const tile_plan_t *)plan, token, slot, "fifo-recv",
                         recv->transition_id);
        return 0u;
    }
}

static inline void maps_fifo_init_tile(const fifo_tile_plan_t *plan,
                                       idma_controller_t *idma_ctrl,
                                       eu_controller_t *eu_ctrl)
{
    for (uint32_t i = 0; i < plan->num_init_l2_reads; ++i)
        issue_l2_read_token((const tile_plan_t *)plan, &plan->init_l2_reads[i], 0u, 0u,
                            idma_ctrl, eu_ctrl);
}

static inline void maps_fifo_run_tile_token(const fifo_tile_plan_t *plan, uint32_t token,
                                            idma_controller_t *idma_ctrl,
                                            eu_controller_t *eu_ctrl)
{
    uint32_t slot = maps_token_slot((const tile_plan_t *)plan, token);
    uint32_t token_start = maps_read_cycle();

    for (uint32_t i = 0; i < plan->num_l2_reads; ++i) {
        uint32_t step_start = maps_read_cycle();
        issue_l2_read_token((const tile_plan_t *)plan, &plan->l2_reads[i], token, slot,
                            idma_ctrl, eu_ctrl);
        maps_trace_duration((const tile_plan_t *)plan, token, slot, "l2-read", i,
                            maps_read_cycle() - step_start);
    }
#if MAPS_FIFO_INPLACE_RECV
    uint8_t inplace[plan->num_recvs + 1u];
    uint32_t held[MAPS_MAX_SLICE_ALIASES];
    uint32_t num_held = 0u;
    maps_fifo_plan_inplace(plan, inplace);
#endif
    for (uint32_t i = 0; i < plan->num_recvs; ++i) {
        uint32_t step_start = maps_read_cycle();
#if MAPS_FIFO_INPLACE_RECV
        if (maps_fifo_wait_recv(plan, &plan->recvs[i], token, slot, inplace[i],
                                idma_ctrl, eu_ctrl, 0))
            held[num_held++] = plan->recvs[i].producer_idx;
#else
        maps_fifo_wait_recv(
            plan, &plan->recvs[i], token, slot, 0u, idma_ctrl, eu_ctrl, 0);
#endif
        maps_trace_duration((const tile_plan_t *)plan, token, slot, "recv",
                            plan->recvs[i].transition_id,
                            maps_read_cycle() - step_start);
    }
    for (uint32_t i = 0; i < plan->num_ops; ++i) {
        uint32_t step_start = maps_read_cycle();
        if (maps_execute_operation((const tile_plan_t *)plan, &plan->ops[i], slot,
                                   plan->operation_runtime) != 0)
            maps_trap();
        maps_trace_duration((const tile_plan_t *)plan, token, slot, "op", i,
                            maps_read_cycle() - step_start);
    }
#if MAPS_FIFO_INPLACE_RECV
    maps_fifo_release_inplace(plan, held, num_held);
#endif
    for (uint32_t i = 0; i < plan->num_sends; ++i) {
        uint32_t step_start = maps_read_cycle();
        maps_fifo_issue_send(plan, &plan->sends[i], token, slot, idma_ctrl, eu_ctrl);
        maps_trace_duration((const tile_plan_t *)plan, token, slot, "send",
                            plan->sends[i].transition_id,
                            maps_read_cycle() - step_start);
    }
    for (uint32_t i = 0; i < plan->num_l2_writes; ++i) {
        uint32_t step_start = maps_read_cycle();
        issue_l2_write_token((const tile_plan_t *)plan, &plan->l2_writes[i], token, slot,
                             idma_ctrl, eu_ctrl);
        maps_trace_duration((const tile_plan_t *)plan, token, slot, "l2-write", i,
                            maps_read_cycle() - step_start);
    }

    uint32_t token_end = maps_read_cycle();
    maps_trace_duration((const tile_plan_t *)plan, token, slot, "token", token,
                        token_end - token_start);
}

static inline void maps_fifo_run_tile_tokens(const fifo_tile_plan_t *plan, uint32_t num_tokens,
                                             idma_controller_t *idma_ctrl,
                                             eu_controller_t *eu_ctrl)
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
#if MAPS_FIFO_INPLACE_RECV
    uint8_t inplace[plan->num_recvs + 1u];
    uint32_t held[MAPS_MAX_SLICE_ALIASES];
    maps_fifo_plan_inplace(plan, inplace);
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
        maps_fifo_finish_slot_sends(plan, eu_ctrl, &states[slot]);

        for (uint32_t i = 0; i < plan->num_l2_reads; ++i) {
            uint32_t step_start = maps_read_cycle();
            issue_l2_read_token((const tile_plan_t *)plan, &plan->l2_reads[i], token, slot,
                                idma_ctrl, eu_ctrl);
            maps_trace_duration((const tile_plan_t *)plan, token, slot, "l2-read", i,
                                maps_read_cycle() - step_start);
        }

        maps_fifo_reap_completed_sends(plan, states);
#if MAPS_FIFO_INPLACE_RECV
        uint32_t num_held = 0u;
#endif
        for (uint32_t i = 0; i < plan->num_recvs; ++i) {
            uint32_t step_start = maps_read_cycle();
#if MAPS_FIFO_INPLACE_RECV
            if (maps_fifo_wait_recv(plan, &plan->recvs[i], token, slot, inplace[i],
                                    idma_ctrl, eu_ctrl, states))
                held[num_held++] = plan->recvs[i].producer_idx;
#else
            maps_fifo_wait_recv(plan, &plan->recvs[i], token, slot, 0u,
                                idma_ctrl, eu_ctrl, states);
#endif
            maps_trace_duration((const tile_plan_t *)plan, token, slot, "recv",
                                plan->recvs[i].transition_id,
                                maps_read_cycle() - step_start);
        }
        maps_fifo_reap_completed_sends(plan, states);
        for (uint32_t i = 0; i < plan->num_ops; ++i) {
            uint32_t step_start = maps_read_cycle();
            if (maps_execute_operation((const tile_plan_t *)plan, &plan->ops[i], slot,
                                       plan->operation_runtime) != 0)
                maps_trap();
            maps_trace_duration((const tile_plan_t *)plan, token, slot, "op", i,
                                maps_read_cycle() - step_start);
        }
#if MAPS_FIFO_INPLACE_RECV
        maps_fifo_release_inplace(plan, held, num_held);
#endif
        maps_fifo_reap_completed_sends(plan, states);
        for (uint32_t i = 0; i < plan->num_l2_writes; ++i) {
            uint32_t step_start = maps_read_cycle();
            issue_l2_write_token((const tile_plan_t *)plan, &plan->l2_writes[i], token, slot,
                                 idma_ctrl, eu_ctrl);
            maps_trace_duration((const tile_plan_t *)plan, token, slot, "l2-write", i,
                                maps_read_cycle() - step_start);
        }

        maps_fifo_reap_completed_sends(plan, states);
        for (uint32_t i = 0; i < plan->num_sends; ++i) {
            const fifo_send_desc_t *send = &plan->sends[i];

            /* tail does not advance until publication, so a second reservation
             * on the same producer sub-ring must wait for the first. */
            maps_fifo_finish_subring_sends(
                plan, eu_ctrl, states, send->dst_hartid, send->producer_idx);

            for (;;) {
                maps_fifo_reap_completed_sends(plan, states);
                maps_fifo_pending_send_t *record =
                    maps_fifo_alloc_pending_send(pending);
                if (record == 0) {
                    if (!maps_fifo_finish_one_send(plan, eu_ctrl, states))
                        maps_trap();
                    continue;
                }

                int rc = maps_fifo_start_send(
                    plan, send, token, slot, idma_ctrl, eu_ctrl, record);
                if (rc == FIFO_ASYNC_STARTED) {
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
                    maps_fifo_finish_all_sends(plan, eu_ctrl, states);
                    uint32_t step_start = maps_read_cycle();
                    maps_fifo_issue_send(
                        plan, send, token, slot, idma_ctrl, eu_ctrl);
                    maps_trace_duration((const tile_plan_t *)plan, token, slot,
                                        "send", send->transition_id,
                                        maps_read_cycle() - step_start);
                    break;
                }
                if (rc == FIFO_ASYNC_IDMA_FULL) {
                    maps_fifo_reap_completed_sends(plan, states);
                    if (!maps_fifo_finish_one_send(plan, eu_ctrl, states))
                        maps_trap();
                    continue;
                }
                if (rc == FIFO_ASYNC_FIFO_FULL) {
                    maps_fifo_finish_all_sends(plan, eu_ctrl, states);
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
    }
    maps_fifo_finish_all_sends(plan, eu_ctrl, states);

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
