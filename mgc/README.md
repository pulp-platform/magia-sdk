# mgc

mgc ("magic") is a Python-syntax language for MAGIA mesh tests. `mgcc.py` compiles a `.mgc`
file to a C test that uses mglib (`mg_idma_*`, `mg_redmule_*`, `mg_event_*`) and fsync.

The compiler handles tile partitioning, L1 layout, DMA address/stride arithmetic, multi-buffer
rotation, event storage, the RedMulE enqueue/commit/start sequence and neighbour-tile addressing.
`.mgc` files are parsed with Python's `ast` module and never executed. The compiler is built on
[xDSL](https://xdsl.dev).

## Setup and usage

```sh
python3 -m venv mgc/.venv
mgc/.venv/bin/pip install -r mgc/requirements.txt

# compile (the header defaults to include/test.h next to the source)
mgc/.venv/bin/python mgc/mgcc.py tests/magia/mesh/mm_os_mgc/mm_os.mgc \
    -o tests/magia/mesh/mm_os_mgc/src/test.c

# build and run as any other test
make build tiles=2 test=test_mm_os_mgc
make run tiles=2 test=test_mm_os_mgc platform=gvsoc

# compiler tests
mgc/.venv/bin/python -m unittest discover -s mgc/tests
```

| Option | Effect |
|---|---|
| `-o FILE` | output file (default: stdout) |
| `--header FILE` | test header with sizes and L2 arrays |
| `--print-ir-after=STAGE` | print the IR after `frontend` or a pass, then stop |
| `--print-ir-after-all` | print the IR after every stage to stderr |
| `--no-format` | skip clang-format (otherwise taken from `PATH` or `llvm/install/bin`) |

Generated files are committed next to their source. Regenerate them after editing a `.mgc` file
or the compiler; the unit tests fail if they are out of date.

## Example tests

| Directory | Source | Content |
|---|---|---|
| `tests/magia/mesh/mm_os_mgc` | `mm_os.mgc`, `mm_os_explicit.mgc` | output-static GEMM, 2-stage pipeline |
| `tests/magia/mesh/mm_ws_mgc` | `mm_ws.mgc`, `mm_ws_explicit.mgc` | weight-static systolic GEMM, partial sums move down columns |
| `tests/magia/mesh/mm_is_mgc` | `mm_is.mgc` | input-static systolic GEMM, partial sums move east along rows |
| `tests/magia/mesh/mm_os_l1k` | `mm_os_l1k.mgc`, `src/kernels.c` | output-static GEMM with a CV32 kernel in the pipeline |

`*_explicit.mgc` files write the schedule by hand; the others use `pipeline()`. The hand-written
references are `mm_os_mglib`, `mm_ws_mglib` and `mm_is_mglib`; the last two are mglib ports of
`mm_ws_2` and `mm_is_2` with the same schedule.

Weight-static, complete per-tile code:

```python
for y_id, x_id in tiles():
    tile_h = y_id.split(N)
    tile_w = x_id.split(K)
    timeslots: u8 = 4
    t_size: u8 = M // timeslots
    t_start: u8 = y_id * 2
    total_timeslots: u8 = (MESH_Y_TILES - 1) * 2 + timeslots + 1

    w = L1.alloc(W[tile_h, tile_w])
    x = L1.multi_buffer(X[0:t_size, tile_h], depth=2)
    y = L1.multi_buffer(Y[0:t_size, tile_w], depth=3)

    for z in range(N_ITERATIONS):
        dma.load(w, W[tile_h, tile_w]).wait()
        for pt in pipeline(timeslots, skew=t_start, steps=total_timeslots, sync=GLOBAL):
            if y_id == 0:
                dma.load(y[pt], Y[pt * t_size:(pt + 1) * t_size, tile_w])
            dma.load(x[pt], X[pt * t_size:(pt + 1) * t_size, tile_h])
            redmule.gemm(x[pt], w, y[pt])                 # y += x @ w
            if y_id == MESH_Y_TILES - 1:
                dma.store(Y[pt * t_size:(pt + 1) * t_size, tile_w], y[pt])
            else:
                dma.store(y.on(1, 0)[pt], y[pt])          # same slot, tile below

    sync(COL)
    if y_id == MESH_Y_TILES - 1:
        check(Y[:, tile_w], Z[:, tile_w], tol=0x11)
```

## Language

### Module level

| Statement | Meaning |
|---|---|
| `A, B = sizes("A_SIZE", "B_SIZE")` | sizes `#define`d in the header (checked) |
| `X = l2("x_inp", (A, B), fp16)` | row-major L2 tensor; the array must exist in the header |
| `NAME = define(value)` | emits `#define NAME value` (`WAIT_MODE` defaults to `WFE`) |
| `k = l1_kernel("c_fn", operands=("dst:out", "src:in"), params=("src.size",), executor=CORE)` | software kernel on L1 data |
| `@test("name", authors=[...])` + `def main():` | the test; the docstring of `main` becomes the C header comment |

### Loops

`main` contains exactly one `for y_id, x_id in tiles():`. Its body runs on every tile (SPMD).
`y_id` and `x_id` can be used in expressions and conditions.

| Loop | Kind | Emitted |
|---|---|---|
| `for y_id, x_id in tiles():` | spatial | tile coordinates from `hartid`, `l1_tile_base` |
| `for v in range(n)`, `range(a, b)` | time | `for` loop; counter type from the bound (`uint8_t` if it fits) |
| `for v in pipeline(n, ...)` | time | software pipeline (see below) |
| `for c in cores():` | intra-tile | reserved, compile error |

Other statements: assignment, `if`/`elif`/`else`, `continue`, `pass`. Docstrings become `/** */`
comments; standalone `#` lines and `comment("...")` become `//` comments.

### Scalars and tile splits

- `name: u8 = expr` declares a variable of type `u8`, `u16`, `u32` or `i32`. Values known at
  compile time are range-checked. An unannotated first assignment declares a `uint32_t`; later
  assignments (`pt = pt + 1`, `pt += 1`) update it. `//` is C integer division.
- `h = y_id.split(E)` splits extent `E` over the mesh rows (`x_id.split` over columns) in blocks
  of `h_max = ceil(E / MESH_Y_TILES)`. Edge tiles are clipped, and tiles with an empty block
  return 0. `h` can index a tensor dimension (start `h_max * y_id`, size `h`). `h.start` and
  `h.size` are expressions.

### Views and L1 buffers

`T[i, j, ...]` on an L2 tensor is a view. Each index is a slice `a:b` or `:` with unit step, a
tile split, or an integer expression (which removes the dimension).

| Expression | Meaning |
|---|---|
| `b = L1.alloc(view)` | dense L1 buffer with the view's shape; the view also fixes the transfer geometry |
| `b = L1.multi_buffer(view, depth=N)` | N slots; `b[c]` for a constant slot, `b[v + k]` with \|k\| < N for a rotating one |
| `b.on(dy, dx)[k]` | slot `k` of `b` on the tile at `(y_id + dy, x_id + dx)`; only as a `dma.store` destination |

### Transfers and events

| Statement | Meaning |
|---|---|
| `e = dma.load(slot, view)` | asynchronous L2 → L1 |
| `e = dma.store(view, slot)` | asynchronous L1 → L2 |
| `e = dma.store(b.on(dy, dx)[k], b[k])` | asynchronous L1 → neighbour's L1, same slot |
| `e.wait()` | wait for the transfer |

Allowed transfers follow `devices.IDMA`. After merging contiguous dimensions, a transfer is at
most `max_rank`-dimensional (currently 2: `mg_idma_memcpy_1d/2d`), with unit steps and a dense
L1 side. L1→L1 within a tile and L2→L2 are rejected.

Each L1 buffer has one event per direction (`idma_evt_<buf>_in/_out`). The two are merged into
`idma_evt_<buf>` when they are never pending at the same time. These are compile errors:
issuing a transfer while the previous one on the same event is certainly still pending, and
waiting on an event that was never issued.

### Accelerators, kernels, synchronization

| Statement | Emitted |
|---|---|
| `j = redmule.gemm(x, w, y)` then `j.wait()` | `mg_redmule_gemm` (y += x @ w), `mg_redmule_wait` |
| `ev = redmule.events(n)` | event array; index with `ev[c]` or `ev[v + k]` |
| `redmule.enqueue(x, w, y, event=ev[k])` | `mg_redmule_gemm_enqueue` |
| `redmule.commit()`, `start()`, `commit_start()` | `mg_redmule_gemm_commit`, `_start`, `_commit_start` |
| `ev[k].wait()` | `mg_redmule_wait` |
| `k(a, b, ...)` (an `l1_kernel`) | `c_fn(a, b, ..., params)`, a synchronous call on the CV32 core |
| `sync(GLOBAL)`, `sync(ROW)`, `sync(COL)` | `fsync_sync_global/row/col` + `eu_fsync_wait` |
| `rotate(v)` | places the multi-buffer slot selection for `v` here (optional) |
| `check(result, golden, tol=0x11)` | per-tile comparison loop, `return errors`; allowed inside `if` |

Accelerator operand ranks, parameters (`m, n, k`) and shape constraints come from
`devices.HWPES`.

L1 kernel operands must be L1 buffers or slots, each declared with a role (`in`, `out`,
`inout`); the compiler uses the roles to order operations. `params` are expressions over operand
shapes: `name.shape[i]`, `name.size` (elements) or `name.bytes`. The C function receives the
operand addresses, then the parameters, all as `uint32_t`, and must be provided by the test
(e.g. `src/kernels.c`).

### `pipeline(n, skew=None, steps=None, sync=None)`

The body describes one iteration `i`: transfers and jobs, optionally inside `if` statements that
do not depend on `i`. The compiler:

1. **Assigns stages.** An item is placed one stage after the latest earlier item that writes a
   slot it reads; otherwise it is in stage 0. Examples: load (0) → gemm (1) → store (2).
2. **Checks multi-buffer depth.** Each multi-buffer must have at least as many slots as the
   number of stages its slots are live across.
3. **Generates the schedule.** A prologue runs stage 0 of iteration 0. In loop step `s`,
   stage `k` works on iteration `s + 1 - k`, guarded by `0 <= s + 1 - k < n`. Within a step, the
   issue order is: transfers, then accelerator jobs, then CV32 calls; the waits follow, in issue
   order.

With `skew` and `steps`, the loop runs over `steps` global time steps. The tile is active when
`0 <= t - skew <= last step`. `sync=SCOPE` adds a barrier after the prologue and at the end of
every time step, active or not.

RedMulE uses early enqueue (program the next job during the DMA, commit it behind the current
one) only when all of these hold:

- the pipeline has 2 stages and no skew;
- it contains exactly one job;
- none of the job's operands is written by another tile.

Otherwise each job is issued and waited within its step. Data from a neighbour is only valid
after the barrier, so the job cannot be committed earlier.

The control-flow shape of the generated C follows the hand-written tests, because GCC's output
depends on it:

- 2-stage pipelines use `if (next) {...; wait} else {wait}`;
- deeper pipelines use one guard per stage.

## Compiler

```
.mgc → frontend → legalize-dma → pipeline → job-lower → multi-buffer → l1-layout → event-alloc → emit-c → clang-format
```

| File | Content |
|---|---|
| `mgcc.py` | command line |
| `mgc/frontend.py` | Python AST → `mg` IR, name resolution |
| `mgc/ir.py` | `mg` dialect (ops, attributes, `!mg.buf` type) |
| `mgc/expr.py` | symbolic integer expressions and conditions |
| `mgc/devices.py` | hardware description: iDMA capabilities, HWPE descriptors, executors, sync scopes |
| `mgc/passes/*.py` | one file per pass |
| `mgc/emit_c.py` | C printer |
| `mgc/driver.py` | runs the stages |
| `tests/test_mgc.py` | compiler tests |

IR conventions:

- Control flow, operations and L1 buffers (`!mg.buf` SSA values) are IR.
- Index arithmetic is a symbolic expression (`#mg.expr`) over C names, so the emitted C keeps
  the source expressions.
- Events are referenced by storage name (`#mg.evt`), not as SSA values, because explicit
  schedules reuse and rotate event variables across iterations and branches.

| Pass | Job |
|---|---|
| `legalize-dma` | views → N-dimensional transfers, checked against `devices.IDMA`; base and per-transfer L2 addresses |
| `pipeline` | `mg.pipeline` → explicit guarded schedule: stages, depth check, job protocol |
| `job-lower` | `mg.job` → RedMulE one-shot or CV32 call; parameters and shape checks from the descriptors |
| `multi-buffer` | slot indices → static addresses or rotating pointers with an `if`/`switch` selection |
| `l1-layout` | identical L1 layout on all tiles; uses `*_max` tile sizes when buffers are accessed remotely; neighbour L1 bases |
| `event-alloc` | event variable names; dataflow check of issue/wait |
| `emit-c` | prints C; each buffer's `len_/std_/reps_/obi_addr_/axi_addr_` is declared once and referenced by name |

Extending:

| To add | Change |
|---|---|
| an iDMA mode (3-D, ...) | `IDMA.max_rank` and `IDMA.functions` in `devices.py` |
| an HWPE accelerator | an `Hwpe` entry in `devices.py` (operands, parameters, constraints, queue depth, function names) and its mglib wrapper |
| a software kernel | `l1_kernel(...)` in the `.mgc` file and the C function |
| Spatz / PULP offload | an asynchronous executor (needs an mglib layer first) |

## Results

2×2 mesh, gvsoc. Exit cycle of each tile, sorted. Every test reports 0 errors.

| Test | Cycles |
|---|---|
| `mm_os_mglib` (reference) | 355929 / 355970 / 391240 / 391241 |
| `mm_os_mgc` | 355950 / 355956 / 391247 / 391255 |
| `mm_ws_2` (raw HAL) | 42954 / 43002 / 153104 / 177774 |
| `mm_ws_mglib` (reference) | 45735 / 45769 / 156153 / 180870 |
| `mm_ws_mgc` (`mm_ws.mgc`) | 45585 / 45603 / 152663 / 177604 |
| `mm_ws_mgc` (`mm_ws_explicit.mgc`) | 45398 / 45432 / 152476 / 177424 |
| `mm_is_2` (raw HAL) | 43292 / 43297 / 171430 / 171435 |
| `mm_is_mglib` (reference) | 46064 / 46064 / 174466 / 174466 |
| `mm_is_mgc` | 44645 / 44668 / 172983 / 173007 |
| `mm_os_l1k` | 466772 / 466787 / 506252 / 506261 |

- **OS:** `mm_os_mglib` clips `tile_w` against `N_SIZE` instead of `K_SIZE`; mgc uses `K_SIZE`.
  With that line changed back, `main` disassembles identically and the cycles are equal.
  `mm_os.mgc` and `mm_os_explicit.mgc` produce the same C.
- **WS/IS:** the mglib references are about 3k cycles slower than the raw-HAL tests (mglib
  event bookkeeping). The mgc versions are 0.8–1.9% faster than the references on the slowest
  tile. The differences:
  - neighbour addresses are computed as `l1_base_s + (slot - l1_tile_base)` instead of a
    per-slot `switch`;
  - the pipelined version prefetches the top row's bias block one step early;
  - L1 offsets use `*_max` tile sizes. `mm_ws_2` and `mm_is_2` use the local tile size, which
    gives wrong neighbour addresses when edge tiles are clipped.
- **L1 kernel:** issuing RedMulE before the CV32 call takes 29k cycles (5.7%) less than calling
  the kernel first.

## Limitations

- One `tiles()` loop and one `pipeline()` per test; `cores()` is not implemented.
- RedMulE early enqueue is used only in 2-stage pipelines.
- fp16 only; RedMulE GEMM is the only accelerator described.
- L1 kernels run synchronously on the CV32 core.
- Comments inside a `pipeline()` body are dropped, except the leading docstring.
- No L1 capacity check (tile sizes are run-time values).
- The event check accepts code that is valid on at least one path, so it can miss errors that
  depend on correlated conditions.
- The generated tests do not support `STALLING=1`.
