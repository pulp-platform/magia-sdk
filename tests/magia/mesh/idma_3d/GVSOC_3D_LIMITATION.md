# `test_idma_3d` and the GVSoC 2D-only iDMA limitation

## Summary

`idma_memcpy_3d` (native 3D IDMA burst) and its test `test_idma_3d` are implemented and
build cleanly, but **the test is intentionally NOT registered in the CI pipeline
(`.gitlab-ci.yml`) yet**, because the GVSoC functional model of the MAGIA iDMA does not
support 3-dimensional transfers. CI runs both RTL and GVSoC jobs; the GVSoC jobs for a 3D
transfer would fail regardless of the SDK code being correct.

## What was added

- `hal/include/idma.h`, `hal/src/idma.c` — `idma_memcpy_3d` prototype, API-struct pointer,
  weak stub, and vtable entry (mirrors `idma_memcpy_2d`).
- `drivers/idma32/src/idma32.c` — `idma32_memcpy_3d` for both the ISA (`IDMA_MM==0`) and
  memory-mapped paths, plus the alias and driver vtable entry. It drives the pre-existing
  `std3`/`rep3` registers (`idma_set_std3_rep3_in/out`, `idma_mm_set_std3_rep3` in
  `targets/magia_v2/include/utils/idma_isa_utils.h`), which `idma_memcpy_2d` already wrote
  with the degenerate value `(0, 0, 1)`.
- `tests/magia/mesh/idma_3d/` — a DMA loopback correctness test (load L2→L1, write back
  L1→L2, verify `y_inp == z_out`) that describes each tile's region as a 3D copy: `tile_h`
  rows split into `reps3` planes of `reps2` rows.
- `tests/magia/mesh/CMakeLists.txt` — `add_subdirectory(idma_3d)`.

## The GVSoC limitation (root cause)

Default MAGIA builds use `idma_mm=1`, so transfers go through the memory-mapped iDMA
controller modeled by
`gvsoc/pulp/pulp/chips/magia_v2/idma_mm_ctrl/idma_mm_ctrl.cpp`.

That model declares and stores the 3rd-dimension registers
(`src_stride_3_reg`, `dst_stride_3_reg`, `reps3_reg` for both DMA engines) and even echoes
them back on MMIO reads — but its transfer-trigger FSM **only reads the 2nd-dimension
registers** (`reps2_reg`, `src_stride_2_reg`, `dst_stride_2_reg`) when it offloads the copy
to the Snitch DMA. Every `.get()` on a dim-3 register is in an MMIO read-back branch, never
in the copy path. See the `dmrep`/`dmstr` offload sequence in `iDMA_mm_ctrl::req`
(both the dir=0 / AXI2OBI and dir=1 / OBI2AXI branches).

Underneath, the Snitch DMA itself uses a **2D middle-end**
(`gvsoc/pulp/pulp/idma/snitch_dma.cpp` includes `me/idma_me_2d.hpp`), so even if the
controller offloaded a 3rd dimension, there is no 3D middle-end to execute it.

### Observed symptom

Running `make run platform=gvsoc test=test_idma_3d tiles=1` reports
`Number of errors: 3072`. For tiles=1 the tile is the full 96×64 matrix; with
`reps3 = 2, reps2 = 48`, `3072 = reps2 × tile_w = 48 × 64` — i.e. exactly the second plane
was never copied. This is the signature of the 3rd dimension being dropped, confirming the
model limitation rather than an SDK bug.

## Why it can't be fixed inside magia-sdk for CI

`gvsoc/` is **not** tracked in this repo. CI obtains it via `make gvsoc_init`, which clones
`github.com/gvsoc/gvsoc.git` and checks out pinned commits (`GVSOC_COMMIT`,
`GVSOC_PULP_COMMIT`, `GVSOC_CORE_COMMIT` from `scripts/deps.env`). Any local edit to the
GVSoC model is overwritten by that fresh clone and never reaches CI.

Enabling 3D on GVSoC therefore requires, upstream in the gvsoc repo:
1. A 3D middle-end (e.g. `idma_me_3d.hpp`) for the Snitch DMA, and
2. Extending `idma_mm_ctrl.cpp` to offload the `stride_3`/`reps_3` dimension,
3. Then bumping the pinned GVSoC commits in `scripts/deps.env`.

RTL is unaffected: the real iDMA implements the ND extension, and the `std3`/`rep3` ISA and
MMIO register layout already exist and are symmetric to `std2`/`rep2`.

## Re-enabling in CI later

Once GVSoC gains 3D iDMA support (and the pinned commits are bumped), add `test_idma_3d`
right after `test_idma_2d, test_idma_1d` in the four `TESTNAME` matrices in
`.gitlab-ci.yml` (`magia_v2_tiles_4_rtl_sim`, `magia_v2_tiles_4_gvsoc_sim`,
`magia_v2_tiles_1_rtl_sim`, `magia_v2_tiles_1_gvsoc_sim`).

If RTL-only CI coverage is desired sooner, `test_idma_3d` can be added to just the two
`*_rtl_sim` matrices; it should pass there and would validate the native 3D burst on real
hardware.
