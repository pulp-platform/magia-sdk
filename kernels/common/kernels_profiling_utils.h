#ifndef KERNELS_PROFILING_UTILS_H_
#define KERNELS_PROFILING_UTILS_H_

#include <stdint.h>

#include "tile.h"

// #define ENABLE_NODE_PROFILING

#ifdef ENABLE_NODE_PROFILING

#define PROF_N_TILES     (MESH_X_TILES * MESH_Y_TILES)
#define PROF_NODE_FIELDS 7

/* Phase accumulators, defined in the deployment template main.c. They live in
 * the per-tile .tile_bss (L1), which is private to each tile and zeroed by
 * crt0, so an update is a local access instead of the NoC round trip that
 * shared-L2 globals would pay on every phase. Being per-tile, they are plain
 * scalars: no hart index, and no risk of one tile clobbering another. */
extern uint32_t prof_cyc_alloc;
extern uint32_t prof_cyc_data_in;
extern uint32_t prof_cyc_compute;
extern uint32_t prof_cyc_data_out;
extern uint32_t prof_cyc_sync;
extern uint32_t prof_cyc_prep;

/* Per-node snapshot table, defined and sized in the deployment template main.c.
 * Unlike the counters this one has to stay in shared L2, since a single tile
 * reads every tile's snapshots to print the report.
 * Flat layout: [node][tile][field]. */
extern uint32_t prof_node[];
#define PROF_NODE_AT(idx, h) (&prof_node[(((idx) * PROF_N_TILES) + (h)) * PROF_NODE_FIELDS])

/* Run `call`, assign its result to `ret`, and add the cycles it took to `acc`. */
#define PROF_PHASE(acc, ret, call)                                                                 \
    do {                                                                                           \
        uint32_t _prof_t0 = perf_get_cycles();                                                     \
        (ret)             = (call);                                                                \
        (acc) += perf_get_cycles() - _prof_t0;                                                     \
    } while (0)

/* Same as PROF_PHASE for calls that return void. */
#define PROF_PHASE_VOID(acc, call)                                                                 \
    do {                                                                                           \
        uint32_t _prof_t0 = perf_get_cycles();                                                     \
        (call);                                                                                    \
        (acc) += perf_get_cycles() - _prof_t0;                                                     \
    } while (0)

/* Snapshot the running counters at a node boundary. A node's cost is the
 * difference between its snapshot and the next one, so one marker per node is
 * enough (plus a final one to close the last node) and no state is needed.
 * The timestamp is stored last, so the six stores above it do not land inside
 * the node's own measurement. */
#define PROF_NODE(idx)                                                                             \
    do {                                                                                           \
        uint32_t *_prof_s = PROF_NODE_AT(idx, get_hartid());                                       \
        _prof_s[1]        = prof_cyc_alloc;                                                        \
        _prof_s[2]        = prof_cyc_data_in;                                                      \
        _prof_s[3]        = prof_cyc_compute;                                                      \
        _prof_s[4]        = prof_cyc_data_out;                                                     \
        _prof_s[5]        = prof_cyc_sync;                                                         \
        _prof_s[6]        = prof_cyc_prep;                                                         \
        _prof_s[0]        = perf_get_cycles();                                                     \
    } while (0)

#else

#define PROF_PHASE(acc, ret, call) ((ret) = (call))
#define PROF_PHASE_VOID(acc, call) (call)
#define PROF_NODE(idx)                                                                             \
    do {                                                                                           \
    } while (0)

#endif /* ENABLE_NODE_PROFILING */

#endif /* KERNELS_PROFILING_UTILS_H_ */
