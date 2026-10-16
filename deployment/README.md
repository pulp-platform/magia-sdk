# MAGIA Deployment

This directory contains the flow that deploys a full **neural network** onto MAGIA. It is
built on top of [Deeploy](https://github.com/pulp-platform/Deeploy): Deeploy parses the
network graph, allocates the buffers and, node by node, emits a C call to a MAGIA kernel.

MAGIA integrates several accelerators. The kernels currently implemented offload their
compute to the **Spatz** vector accelerator, but the flow is not tied to Spatz: both the
kernels and the tests follow a `<operator>/<data-format>/<arch>/` directory layout, so
additional data formats (beyond `fp16`) and additional execution targets (beyond `spatz`)
can be added later. Today `<data-format>/<arch>` is always `fp16/spatz`.

- `MagiaDeeployTarget/` — the MAGIA Deeploy target (platform, bindings, templates,
  code-transformation passes).
- `generate_with_spatz.py` + `main.c` — generator and CV32 host `main` template for the
  **CV32 + Spatz** path.
- `generate.py` + `test.c` — generator and host `main` template for the **CV32-host-only**
  path (no accelerator offload).
- `tests/<operator>/<data-format>/<arch>/` — one folder per unit test / network, each with
  a `generate_network.py`. Our tests are described in ONNX: the script produces
  `network.onnx`, `inputs.npz` and `outputs.npz` (the golden reference computed with
  `onnxruntime` on the fp16 model).

## Deploy commands

Both rules live in the top-level `Makefile`:

- **`make deploy_with_spatz test=<name>/<fmt>/<arch> platform=<rtl|gvsoc> tiles=<N>`**
  The **CV32 + Spatz** path. Generates the network C code with `generate_with_spatz.py`
  into `tests/spatz_on_magia/deeploy_<name>_<fmt>_<arch>/`, builds the CV32 executable with
  the embedded Spatz binary, and runs it. This is the path used for all the networks here.

- **`make deploy test=<name> platform=<rtl|gvsoc> tiles=<N>`**
  The **CV32-host-only** path: generates with `generate.py` and runs on the plain CV32 host,
  without offloading to an accelerator.

Example:

```bash
make deploy_with_spatz test=resnet-reduced/fp16/spatz platform=gvsoc tiles=2
```

## Adding a new operator

The frontend (turning a graph node into buffers and an *operator representation*, i.e. the
`parser` / `type-checker` / `layer`) is provided by Deeploy and reused. To make a new
operator `Op` deployable on MAGIA, add the following **artifacts** (paths
use the current `fp16/spatz` layout):

**1. Accelerator task** — `kernels/<op>/fp16/spatz/spatz_task/<op>_fp16_spatz_task.c`
The vector code that runs on the accelerator; entry point `<op>_fp16_spatz_task()`. It
reads its shard from the params struct in L1 and writes the result back.

**2. CV32 host kernel** — `kernels/<op>/fp16/spatz/src/<op>_fp16_spatz.c`
`MAGIA_<op>_fp16_spatz(...)`: shards the tensors across tiles (see *Shard strategy* in the
table below), fills the params struct and offloads the task to the accelerator.

**3. Host header** — `kernels/<op>/fp16/spatz/include/<op>_fp16_spatz.h`
The prototype of `MAGIA_<op>_fp16_spatz(...)`, e.g.
`void MAGIA_relu_fp16_spatz(const float16 *X, float16 *Y, uint32_t size);`

**4. Params struct** — `kernels/<op>/fp16/spatz/include/<op>_fp16_spatz_params.h`
The struct shared between host and accelerator (shard pointers, start/end/len, extra
attributes).

**5. Template** — `MagiaDeeployTarget/Templates/`
A `NodeTemplate`: `alignToContext()` looks up the buffers and sets any extra operator
representation keys; `referenceTemplate` emits the C call, e.g.
`MAGIA_<op>_fp16_spatz(${data_in}, ${data_out}, ${size});`. The `${...}` keys are the ones
the frontend `parser` puts into the operator representation.

**6. Binding + wiring**
- In the bindings, add a binding that pairs the operator's type-checker (input/output
  types) with the template.
- In the platform, create the operator's mapper (parser + binding) and add an entry to the
  operator map that associates the operator name with its layer (built from the mapper).

**7. Unit test** — `deployment/tests/<op>/fp16/spatz/generate_network.py`
A small script that builds a single-node graph for `Op`, runs it through a reference
runtime to get the golden output, and saves `network.onnx`, `inputs.npz`, `outputs.npz`.
Run it, then `make deploy_with_spatz test=<op>/fp16/spatz ...`.

> Shape-only operators (**Reshape**, **Flatten**, **Split**) have **no kernel**: they are
> handled as zero-copy buffer aliases directly by the Deeploy target (each output points
> into the input buffer), so artifacts 1–4 are not needed for them.

## Supported operators

Cross-checked against the operator map (`Platform.py`), the kernels in `kernels/` and the
unit tests in `deployment/tests/`. Shapes are `[N, C, H, W]`; *Agnostic* means the kernel
only needs the total element count. `HID` = number of harts (tiles). `G` is the number of
groups.

| Operator | Input shape | Shard strategy | Assumptions |
|---|---|---|---|
| Add | Agnostic | size / HID | |
| AveragePool | [N,C,H,W] | (N*C) / HID | no auto_pad; no ceil_mode; symmetric H/W pad |
| BatchNorm | [N,C,H,W] | (N*C) / HID | |
| Ceil | Agnostic | size / HID | |
| Clip | Agnostic | size / HID | |
| Col2Im | [N,C,H,W] | (N*C) / HID | |
| Concat | Agnostic | iterations / HID | axis != 0 |
| Conv | [N,C,H,W] | depthwise: (N*C_out) / HID; otherwise: H_out / HID, or C_out / HID for small outputs | no auto_pad; no dilations; symmetric pad |
| ConvTranspose | [N,C,H,W] | (N*C_out) / HID | no auto_pad; no dilations; no output_padding; no output_shape |
| Div | Agnostic | size / HID | |
| Elu | Agnostic | size / HID | |
| Exp | Agnostic | size / HID | |
| Flatten | — | — | zero-copy buffer alias (no kernel) |
| Floor | Agnostic | size / HID | |
| Gather | Agnostic | iterations / HID | indices.size == 1; axis != 0 |
| Gelu | Agnostic | size / HID | tanh approximation |
| GEMM | 2D matrices A(M,K)/(K,M), B(K,N)/(N,K), C(M,N), Y(M,N) | min(M,N) / HID | transA/transB supported (matrices stored non-transposed in L1) |
| GlobalAvgPool | [N,C,H,W] | (N*C) / HID | |
| GlobalMaxPool | [N,C,H,W] | (N*C) / HID | |
| GroupNorm | [N,C,H,W] | (N*G) / HID | |
| HardSigmoid | Agnostic | size / HID | |
| HardSwish | Agnostic | size / HID | |
| InstanceNorm | [N,C,H,W] | (N*C) / HID | |
| LayerNorm | Agnostic | iterations / HID | axis = -1 (forced by the frontend) |
| LeakyRelu | Agnostic | size / HID | |
| MatMul | [N,C,H,W] | (N*C) / HID | |
| MaxPool | [N,C,H,W] | (N*C) / HID | no auto_pad; no ceil_mode (floor only); symmetric H/W pad |
| Mul | Agnostic | size / HID | |
| ReduceMean | Agnostic | iterations / HID | |
| Relu | Agnostic | size / HID | |
| Reshape | — | — | zero-copy buffer alias (no kernel) |
| Resize | [N,C,H,W] | (N*C) / HID | Resize Nearest-Neighbor only |
| ScatterElements | [N,C,H,W] | iterations / HID | |
| Selu | Agnostic | size / HID | |
| Sigmoid | Agnostic | size / HID | |
| Slice | Agnostic | iterations / HID | |
| Softmax | rank 2/3/4 [N,C,H,W] | (N*C*H) / HID | any axis |
| Split | — | — | zero-copy buffer alias (no kernel); strided split not supported |
| Sub | Agnostic | size / HID | |
| Swish | Agnostic | size / HID | |
| Tanh | Agnostic | size / HID | |
| Transpose | Agnostic | N_out / HID | perm[0] must be 0 (the kernel itself moves L2→L1 and shards on axis 0) |

Notes:
- **iterations** (Concat, Gather, LayerNorm, ReduceMean, ScatterElements, Slice) = the
  product of the tensor dimensions before the operator's axis (for LayerNorm, all
  dimensions except the last, normalized, one).
- **Elementwise broadcast** (Add / Sub / Mul / Div) is handled by a dedicated
  broadcast layer that expands the operand shapes (numpy semantics), so a scalar/smaller
  operand is materialized to the full output shape.
- **Conv** has two kernels, picked per node by Deeploy: depthwise convolutions (`group` equal to
  the input channels) run the direct `conv2d` kernel, every other convolution runs `conv2dgemm`
  (im2col + GEMM).
- **Fast exponential**: Elu, Exp, Gelu, Selu, Sigmoid, Softmax, Swish and Tanh use a fast
  Schraudolph reinterpret-cast approximation of the exponential (~3% relative error).

## Cycle profiling

Optional per-node cycle profiling, off by default, measured on gvsoc.

```bash
make deploy_with_spatz test=tiny-vit/fp16/spatz platform=gvsoc \
     enable_node_profiling=1 2>&1 | tee run.log
```

`layer_depth` (default 2) sets how many levels of the export hierarchy define a
layer: at 2, `/layers.0/blocks.0/conv1/c/Conv` belongs to `/layers.0/blocks.0`.

### API

Everything lives in `kernels/common/kernels_profiling_utils.h`. Five counters, one
per phase, plain scalars in each tile's own L1 (`.tile_bss`):

```c
prof_cyc_alloc  prof_cyc_data_in  prof_cyc_compute
prof_cyc_data_out prof_cyc_sync
```

and three macros:

| macro | what it does |
|---|---|
| `PROF_PHASE(counter, ret, call)` | runs `ret = call`, adds the cycles it took to `counter` |
| `PROF_PHASE_VOID(counter, call)` | same, for calls returning `void` |
| `PROF_NODE(idx)`                 | snapshots the counters at a node boundary |

`PROF_NODE`, and the `PROF_PHASE_VOID(prof_cyc_sync, ...)` around the tile
barrier, are injected into the generated code, so you never write them by hand.
`PROF_PHASE` is the one that matters when **adding a new operator**: include the header and wrap the phases in the kernel's entry point,
picking the counter that matches each one.

```c
#include "kernels_profiling_utils.h"

void MAGIA_myop_fp16_spatz(const float16 *X, float16 *Y, uint32_t size)
{
    int ret;
    volatile myop_fp16_spatz_params_t *params;

    PROF_PHASE(prof_cyc_alloc, ret, alloc_l1((void **)&params, size));
    PROF_PHASE(prof_cyc_data_in, ret, init_input_params((void *)params, X));
    PROF_PHASE(prof_cyc_compute, ret, offload_spatz_task((void *)params));
    PROF_PHASE(prof_cyc_data_out, ret, store_result((void *)params, Y));
}
```

### Output

The log is **self-contained** (no sidecar file) and **ordered**: the report runs
after the closing barrier and is printed by a single tile, which can read every
tile's numbers because the tables live in shared L2, so lines never interleave.

Every line is prefixed with `[PROF]`, so the whole report is one `grep` away. It
walks the network layer by layer: each layer opens with a header, then comes one
line per node and tile, and the layer closes with its own per-tile recap:

```
[PROF] ### Profiling for Layer 0 (/patch_embed/seq) ###
[PROF] hid=0 layer_idx=0 node_idx=0 node=/patch_embed/seq/seq.0/c/Conv op=Conv node_cycles=13883 alloc=923 in=2744 cmp=8121 out=369 snc=1193
[PROF] hid=1 layer_idx=0 node_idx=0 node=/patch_embed/seq/seq.0/c/Conv op=Conv node_cycles=13911 ...
...
[PROF] Layer 0 recap:
[PROF] hid=0 layer_idx=0 layer_cycles=22138 alloc=1702 in=4012 cmp=12544 out=810 snc=1764
[PROF] hid=1 layer_idx=0 layer_cycles=22176 ...
```

and the run ends with the totals, followed by the usual per-tile cycle count:

```
[PROF] ### Final recap ###
[PROF] hid=0 total_cycles=333471 alloc=41103 in=37646 cmp=152376 out=20799 snc=24781
...
[CV32 (0)] Run completed in 333641 cycles
```

| field | meaning |
|---|---|
| `hid`           | **hart id**, i.e. which tile of the mesh (`0..tiles^2-1`) |
| `layer_idx`     | index of the layer the node belongs to; its name is in the header |
| `node_idx`      | index of the node in execution order |
| `node`          | the original ONNX node name |
| `op`            | the node's operator |
| `node_cycles`   | cycles between this node's marker and the next one — the node's whole cost, sync included |
| `layer_cycles`  | same, summed over the nodes of the layer (recap lines) |
| `total_cycles`  | same, summed over the whole network (final recap) |
| `alloc`         | L1 allocation |
| `in`            | copy-in L2->L1 (iDMA + wait) |
| `cmp`           | Spatz offload (+ wait) |
| `out`           | copy-out L1->L2 (iDMA + wait) |
| `snc`           | tile barrier following the node |
| `other`         | not printed: `node_cycles` minus the five phases above — the generated glue around each kernel call, the zero-copy nodes that run no kernel at all, and the profiling markers themselves |

### Measured results

Measured on gvsoc (`20ccbf1e`, chips-it) with a **2x2 mesh**.

Note these numbers are **indicative**: the networks are deployed **un-tiled**, and
each kernel does its own L2 <-> L1 transfers.

| network | nodes | total | alloc | in | cmp | out | snc | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `tiny-vit` | 189 | 333 450 | 41 734 (12.5%) | 35 737 (10.7%) | 128 722 (38.6%) | 18 265 (5.5%) | 47 686 (14.3%) | 61 306 (18.4%) |
| `resnet-reduced` | 30 | 1 361 040 | 10 336 (0.8%) | 284 505 (20.9%) | 967 249 (71.1%) | 54 940 (4.0%) | 33 223 (2.4%) | 10 787 (0.8%) |
| `bert-reduced` | 64 | 1 204 914 | 12 435 (1.0%) | 70 742 (5.9%) | 258 087 (21.4%) | 28 343 (2.4%) | 815 176 (67.7%) | 20 131 (1.7%) |
| `efficientnet-reduced` | 94 | 1 287 200 | 32 290 (2.5%) | 671 761 (52.2%) | 408 675 (31.7%) | 37 779 (2.9%) | 101 111 (7.9%) | 35 584 (2.8%) |
| `yolo-reduced` | 233 | 9 997 376 | 59 758 (0.6%) | 945 334 (9.5%) | 8 541 188 (85.4%) | 134 440 (1.3%) | 235 556 (2.4%) | 81 100 (0.8%) |

### Optional: turning the log into tables

`scripts/prof_report.py` is a convenience, not a requirement: the log is readable
as it is. It takes the log alone — it recovers the layer and node names from it —
and **prints to stdout** three tables: cycles **per layer**, the **top nodes** by
cycles, and a **per-operator** summary, each with percentages of the total.

```bash
# tables on stdout
scripts/prof_report.py run.log

# 20 nodes ranked, and the raw numbers into a file
scripts/prof_report.py run.log --top 20 --csv nodes.csv
```

Nothing is written unless `--csv` is given, and that is the only file produced:
one row per (node, tile) with `node_idx, node, op, layer_idx, layer, hid`, then
`node_cycles` and the five phases, ready for a spreadsheet. The script prints the
path it wrote. `--agg` chooses how the tiles are collapsed:
`crit` (default) takes every field from the tile that gates the node, keeping the
parts consistent with the total, while `mean` averages them — useful to spot load
imbalance, since a node whose mean is dominated by `snc` means the other tiles
were waiting.
