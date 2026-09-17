/*
==================================================================
 MAGIA Shared-L2-Address Contention Latency Benchmark
==================================================================

 Goal:
   There is exactly ONE value in L2 (y_shared). EVERY core reads
   THAT SAME address and measures its own latency doing so - both
   via a direct load and via IDMA - so you can see whether hitting
   a shared L2 location concurrently from many cores costs more
   than a single core accessing it alone (contention), and whether
   that cost differs between the two access methods.

 Method:
   - Tile 0 writes y_shared once, then a barrier makes sure every
     other core only starts reading after it's actually there.
   - For each of NUM_TRIALS trials, ALL cores hit a barrier first
     (so their bursts of repeated accesses start together and
     actually overlap on the shared address), then each core times
     its own LOAD_REPEATS direct reads, and separately its own
     DMA_REPEATS IDMA transfers, of y_shared.
   - Every core stores its own min/avg/max cycle counts (indexed by
     its own hartid) into L2 result arrays. Tile 0 collects and
     prints the full per-core table at the end.
   - Every read is checked against the known value of y_shared, so
     a wrong answer shows up as a count, not just a latency number.

 Reading the output:
   If there's no real contention, every core's numbers should look
   similar to a single-core baseline. If cores start to disagree
   (some much slower than others) or the average creeps up compared
   to a single-core run, that's contention on the shared L2 address
   / interconnect showing up.

 !! PLATFORM ASSUMPTION FOR THE DIRECT METHOD - PLEASE VERIFY !!
   Reading L2 directly from any core is a much safer assumption than
   the earlier "read another tile's L1 directly" benchmark - L2 is
   generally the architecturally shared memory in these designs. But
   still worth double-checking against your platform docs that plain
   loads to L2 are supported from every core's pipeline (as opposed
   to only being reachable via IDMA), since that's what METHOD A
   (direct) below relies on.
==================================================================
*/

#include <stdint.h>

#include "tile.h"
#include "idma.h"
#include "fsync.h"
#include "eventunit.h"

#include "sme3Da.h"

#define WAIT_MODE      WFE
#define clock_freq_MHz 1000
#define MAX_CORES      128
#define PAYLOAD_BYTES  (PAYLOAD_WORDS * (int)sizeof(int32_t))

/*
==================================================================
 L2 profiling storage - one slot per possible hart, written only by
 that hart (no cross-core races on these arrays; each core owns its
 own index).
==================================================================
*/
static uint32_t HARTIDS[MAX_CORES] __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t NUM_CORES          __attribute__((section(".l2"), aligned(64))) = 0;

static uint32_t direct_min[MAX_CORES]   __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t direct_max[MAX_CORES]   __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t direct_sum[MAX_CORES]   __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t direct_wrong[MAX_CORES] __attribute__((section(".l2"), aligned(64))) = {0};

static uint32_t dma_min[MAX_CORES]      __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t dma_max[MAX_CORES]      __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t dma_sum[MAX_CORES]      __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t dma_wrong[MAX_CORES]    __attribute__((section(".l2"), aligned(64))) = {0};

static uint32_t cyc_to_ns(uint32_t cycles)
{
    return (cycles * 1000u) / clock_freq_MHz;
}

static uint32_t find_max(uint32_t input[], uint32_t length)
{
    uint32_t max = 0;
    for (uint32_t i = 0; i < length; i++) {
        if (input[i] > max) max = input[i];
    }
    return max;
}

static uint32_t find_min(uint32_t input[], uint32_t length)
{
    uint32_t min = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < length; i++) {
        if (input[i] < min) min = input[i];
    }
    return min;
}

int main(void)
{
    uint32_t hartid = get_hartid();

    /*
    ==============================================================
    Initialize IDMA / FSYNC / Event Unit
    ==============================================================
    */
    idma_config_t idma_cfg = { .hartid = hartid };
    idma_controller_t idma_ctrl = { .base = NULL, .cfg = &idma_cfg, .api = &idma_api };
    idma_init(&idma_ctrl);

    fsync_config_t fsync_cfg = { .hartid = hartid };
    fsync_controller_t fsync_ctrl = { .base = NULL, .cfg = &fsync_cfg, .api = &fsync_api };
    fsync_init(&fsync_ctrl);

    eu_config_t eu_cfg = { .hartid = hartid };
    eu_controller_t eu_ctrl = { .base = NULL, .cfg = &eu_cfg, .api = &eu_api };
    eu_init(&eu_ctrl);

    eu_clear_events(0xFFFFFFFF);
    eu_idma_init(&eu_ctrl, 0);
    eu_fsync_init(&eu_ctrl, 0);

    /*
    ==============================================================
    NUM_CORES discovery (same pattern as GeoCSR Phase 1)
    ==============================================================
    */
    HARTIDS[hartid] = 1;
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    if (hartid == 0) {
        for (int i = 0; i < MAX_CORES; i++) {
            if (HARTIDS[i] == 1) {
                NUM_CORES++;
            }
        }
    }

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    if (hartid >= NUM_CORES) {
        return 0;
    }

    perf_start();

    /*
    ==============================================================
    Step 1: tile 0 writes the ONE shared L2 value. Everyone else
    just waits for it.
    ==============================================================
    */
    if (hartid == 0) {
        for (int k = 0; k < PAYLOAD_WORDS; k++) {
            y_shared[k] = (int32_t)(Y_L2_BASE + k);
        }
    }

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    int32_t expected = 0;
    for (int k = 0; k < PAYLOAD_WORDS; k++) {
        expected += (int32_t)(Y_L2_BASE + k);
    }

    uint32_t l1 = get_l1_base(hartid);
    volatile int32_t *recv_buf = (int32_t *)(l1 + RECV_OFFSET);

    /* IMPORTANT: must be a volatile pointer, not a direct index into the
       plain y_shared array. Without this, the compiler can (and, for
       PAYLOAD_WORDS > 1, reliably will) prove the inner k-loop computes
       the same value on every one of the LOAD_REPEATS outer iterations
       and hoist the actual memory reads out entirely - which is exactly
       what produced the fake "1 cycle" results. */
    volatile int32_t *y_ptr = y_shared;

    /*
    ==============================================================
    Step 2: METHOD A - DIRECT. Every core reads y_shared for itself.
    A barrier before each trial lines up all cores' bursts so they
    actually overlap on the shared address.
    ==============================================================
    */
    {
        uint32_t min_c = 0xFFFFFFFFu, max_c = 0, sum_c = 0, wrong = 0;

        for (int trial = 0; trial < NUM_TRIALS; trial++) {

            fsync_sync_global(&fsync_ctrl);
            eu_fsync_wait(&eu_ctrl, WAIT_MODE);

            int32_t sum = 0;
            uint32_t t0 = perf_get_cycles();
            for (int r = 0; r < LOAD_REPEATS; r++) {
                sum = 0;
                for (int k = 0; k < PAYLOAD_WORDS; k++) {
                    sum += y_ptr[0];
                }
            }
            uint32_t total_cycles = perf_get_cycles() - t0;
            uint32_t per_access = total_cycles / LOAD_REPEATS;

            if (per_access < min_c) min_c = per_access;
            if (per_access > max_c) max_c = per_access;
            sum_c += per_access;

            if (sum != expected) wrong++;
        }

        direct_min[hartid]   = min_c;
        direct_max[hartid]   = max_c;
        direct_sum[hartid]   = sum_c;
        direct_wrong[hartid] = wrong;
    }

    /*
    ==============================================================
    Step 3: METHOD B - IDMA. Every core pulls its own copy of
    y_shared into its own L1, timed. Same per-trial barrier so the
    bursts overlap.
    ==============================================================
    */
    {
        uint32_t min_c = 0xFFFFFFFFu, max_c = 0, sum_c = 0, wrong = 0;

        for (int trial = 0; trial < NUM_TRIALS; trial++) {

            fsync_sync_global(&fsync_ctrl);
            eu_fsync_wait(&eu_ctrl, WAIT_MODE);

            uint32_t t0 = perf_get_cycles();
            for (int r = 0; r < DMA_REPEATS; r++) {
                idma_memcpy_1d(&idma_ctrl, DMA_CH_IN, (uint32_t)y_shared,
                               (uint32_t)recv_buf, (uint32_t)PAYLOAD_BYTES);
                eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
            }
            uint32_t total_cycles = perf_get_cycles() - t0;
            uint32_t per_access = total_cycles / DMA_REPEATS;

            if (per_access < min_c) min_c = per_access;
            if (per_access > max_c) max_c = per_access;
            sum_c += per_access;

            int32_t sum = 0;
            for (int k = 0; k < PAYLOAD_WORDS; k++) {
                sum += recv_buf[k];
            }
            if (sum != expected) wrong++;
        }

        dma_min[hartid]   = min_c;
        dma_max[hartid]   = max_c;
        dma_sum[hartid]   = sum_c;
        dma_wrong[hartid] = wrong;
    }

    /* Make sure every core has finished writing its own results
       before tile 0 reads them all. */
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /*
    ==============================================================
    Step 4: tile 0 collects and prints the per-core table.
    ==============================================================
    */
    if (hartid == 0) {

        printf("=== MAGIA Shared-L2-Address Contention Benchmark ===\n");
        printf("NUM_CORES     : %u\n", NUM_CORES);
        printf("PAYLOAD_WORDS : %u (%u bytes), shared L2 address for all cores 0x%x\n",
               (uint32_t)PAYLOAD_WORDS, (uint32_t)PAYLOAD_BYTES, &y_ptr);
        printf("LOAD_REPEATS  : %u, DMA_REPEATS: %u, NUM_TRIALS: %u\n",
               (uint32_t)LOAD_REPEATS, (uint32_t)DMA_REPEATS, (uint32_t)NUM_TRIALS);
        printf("\n");
        printf("core  direct_avg_cyc   idma_avg_cyc\n");

        for (uint32_t c = 0; c < NUM_CORES; c++) {
            uint32_t d_avg = direct_sum[c] / NUM_TRIALS;
            uint32_t m_avg = dma_sum[c]    / NUM_TRIALS;

            printf("%u          %u         %u\n",
                c, d_avg, m_avg);
        }

        /* Quick contention summary: spread between the fastest and
           slowest core's average tells you how much the shared
           address costs some cores relative to others. */
        uint32_t direct_avgs[MAX_CORES], dma_avgs[MAX_CORES];
        for (uint32_t c = 0; c < NUM_CORES; c++) {
            direct_avgs[c] = direct_sum[c] / NUM_TRIALS;
            dma_avgs[c]    = dma_sum[c]    / NUM_TRIALS;
        }

        uint32_t d_best  = find_min(direct_avgs, NUM_CORES);
        uint32_t d_worst = find_max(direct_avgs, NUM_CORES);
        uint32_t m_best  = find_min(dma_avgs, NUM_CORES);
        uint32_t m_worst = find_max(dma_avgs, NUM_CORES);

        printf("\nDIRECT: best-core avg = %u cyc, worst-core avg = %u cyc (spread = %u cyc)\n",
               d_best, d_worst, d_worst - d_best);
        printf("IDMA  : best-core avg = %u cyc, worst-core avg = %u cyc (spread = %u cyc)\n",
               m_best, m_worst, m_worst - m_best);
        printf("(a large spread suggests real contention on the shared L2 address /\n");
        printf(" interconnect; a small spread suggests each core is getting an\n");
        printf(" essentially uncontended path to L2.)\n");

        uint32_t total_wrong = 0;
        for (uint32_t c = 0; c < NUM_CORES; c++) {
            total_wrong += direct_wrong[c] + dma_wrong[c];
        }
        printf("\nTotal wrong-answer trials (both methods, all cores): %u\n", total_wrong);

        return (int)total_wrong;
    }

    return 0;
}