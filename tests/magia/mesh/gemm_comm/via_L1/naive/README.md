# via_l2_naive — GEMM Chain Test (L2 Communication)

4-GEMM chain with task-level and row-parallel data parallelism across a 64-tile (8×8) mesh.
Inter-tile communication happens entirely through shared L2 memory: tiles DMA intermediate results
into L2 buffers that subsequent phases read back.

## Overview

- Phase 1:
    - Move data L2->L1 (GEMM 1 and GEMM 2)
    - GEMM 1 and GEMM 2
    - Move result source L1->target L1 (GEMM 1 and GEMM 2 to GEMM 3)
    - === Global barrier ===

- Phase 2:
    - GEMM 3
    - Move result source L1->target L1 (GEMM 3 to GEMM 4)
    - === Global barrier ===

- Phase 3:
    - Move data L2->L1 (M5 for GEMM 4)
    - GEMM 4
    - Move result L1->L2 (GEMM 4)
    - === Global barrier ===

## Tile Groups

| Group   | Count | Role          |
|---------|-------|---------------|
| GEMM 1  | 4     | Phase 1       |
| GEMM 2  | 24    | Phase 1       |
| GEMM 3  | 12    | Phase 2       |
| GEMM 4  | 24    | Phase 3       |

## Phases

### Phase 0 — Initialization
1. Each tile reads its `hartid`, derives `x_id`, `y_id`, and its L1 base address.
2. iDMA, RedMulE, FractalSync, and EventUnit controllers are initialized.
3. **Global barrier** (FractalSync at `MAX_SYNC_LVL - 1`): all tiles wait until every tile has
   finished BSS zeroing (performed by tile 0 in `crt0.S`) before any tile writes to L2 output
   buffers.

---

### Phase 1 — Parallel GEMMs (GEMM1 ∥ GEMM2)

Both groups run concurrently. Within each group, output rows are split evenly across tiles.

**GEMM1** — `R1[A×C] = M1[A×B] @ M2[B×C]`

For each tile in the group:
1. Compute the tile's row slice (`start_row`, `num_rows`) of the output.
2. DMA `M1[num_rows × B]` from L2 into L1.
3. DMA full `M2[B × C]` from L2 into L1.
4. Zero the L1 accumulator buffer for `R1_slice`.
5. RedMulE GEMM: `R1_slice = M1_slice @ M2`.
6. DMA `R1_slice` to correct positions in L1 of tiles responsible for GEMM 3, where R1 is needed.

**GEMM2** — `R2[C×E] = M3[C×D] @ M4[D×E]`

Same steps as GEMM1, operating on `M3`, `M4` → `r2_out` (`R2_slice = M3_slice @ M4`).

**Global barrier** — all tiles synchronize before Phase 2 reads `r1_out` / `r2_out`.

---

### Phase 2 — GEMM3

**GEMM3** — `R3[A×E] = R1[A×C] @ R2[C×E]`

For each tile in the group:
1. Compute the tile's row slice of `R3`.
2. Zero the L1 accumulator buffer for `R3_slice`.
3. RedMulE GEMM: `R3_slice = R1_slice @ R2`.
4. DMA `R3_slice` to correct positions in L1 of tiles responsible for GEMM 4, where R1 is needed.

**Global barrier** — all tiles synchronize before Phase 3 reads `r3_out`.

---

### Phase 3 — GEMM4

**GEMM4** — `O[A×F] = R3[A×E] @ M5[E×F]`

For each tile in the group:
1. Compute the tile's row slice of `O`.
2. DMA full `M5[E × F]` from L2 into L1.
3. Zero the L1 accumulator buffer for `O_slice`.
4. RedMulE GEMM: `O_slice = R3_slice @ M5`.
5. DMA `O_slice` back to L2 at `o_out[start_row * F]`.

**Global barrier** — all tiles synchronize before validation.

---

### Validation

Tile 0 compares each element of `o_out` against the precomputed `o_golden` array.
Values are compared as integer millis (via `fp16_to_millis`) with an absolute threshold of
`0.008`. Errors and the final count are printed via `printf`.
