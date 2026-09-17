#ifndef DATA_H_
#define DATA_H_

#include <stdint.h>

/*
==================================================================
 Benchmark parameters
==================================================================
 PAYLOAD_WORDS : how many int32_t words make up the shared value
                 every core reads. Default 1. Bump this to see how
                 IDMA vs direct-load behavior (and any contention
                 effect) changes with transfer size.

 Y_L2_BASE     : the shared L2 value is y_shared[k] = Y_L2_BASE + k,
                 identical for every core - there is exactly ONE
                 copy in L2, and every core reads that same address.

 RECV_OFFSET   : byte offset within each core's OWN L1 where the
                 IDMA method lands its copy of y_shared. Each core
                 uses its own L1 for this, so there's no collision
                 between cores on the destination side - only the
                 shared L2 source address is contended.

 LOAD_REPEATS  : direct-access repeats per trial (amortizes the
                 overhead of calling perf_get_cycles() itself).
 DMA_REPEATS   : IDMA-transfer repeats per trial.
 NUM_TRIALS    : independent repeat-blocks per method. A barrier is
                 placed before every trial so all cores start their
                 burst of repeats together, which is what actually
                 creates contention on the shared L2 address instead
                 of each core quietly measuring one at a time.
 DMA_CH_IN     : DMA channel used for "into this tile's L1"
                 transfers (paired with eu_idma_wait_a2o).
==================================================================
*/
#define PAYLOAD_WORDS  10
#define Y_L2_BASE      9000

#define RECV_OFFSET    0

/* NOTE: these must stay > 1 to be meaningful. LOAD_REPEATS/DMA_REPEATS
   amortize the fixed overhead of calling perf_get_cycles() itself, and
   NUM_TRIALS is what lets min/avg/max (and the contention spread) mean
   anything at all. Setting any of these to 1 turns the measurement back
   into a single noisy sample. */
#define LOAD_REPEATS   10
#define DMA_REPEATS    10
#define NUM_TRIALS     5
#define DMA_CH_IN      0

/*
==================================================================
 y_shared - the single, shared L2 location every core reads. Pinned
 to ".l2" like the profiling arrays in the GeoCSR sample. Filled in
 once, by tile 0, at runtime. This is the ONLY location that should
 ever be read in the DIRECT/IDMA loops below - do not introduce a
 second, unwritten global here (see the "p" bug write-up).
==================================================================
*/
static int32_t y_shared[PAYLOAD_WORDS] __attribute__((section(".l2"), aligned(64)));

#endif /* DATA_H_ */