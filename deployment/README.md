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
| Conv | [N,C,H,W] | (N*C_out) / HID | no auto_pad; no dilations; symmetric pad |
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
- **Fast exponential**: Elu, Exp, Gelu, Selu, Sigmoid, Softmax, Swish and Tanh use a fast
  Schraudolph reinterpret-cast approximation of the exponential (~3% relative error).
