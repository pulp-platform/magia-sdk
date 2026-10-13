#!/usr/bin/env bash
# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Run onnx_mobilevit_v2 up to layer N and compare that layer's output against the golden
# tensor the generator left in test_data/layers/.
#
# The full test only has a compiled-in golden for the final logits, so when it fails this
# is how the failing layer gets found - bisect N over 1..205. Layers get more expensive the
# earlier they are (layer 0 alone is ~40 s at 16 tiles), so bisect downward from a
# known-bad N rather than sweeping up from 1.
#
# Usage: ./check_layer.sh <N> [tiles]
#
# Needs: the toolchain on PATH, LLVM_INSTALL_DIR pointing at ./llvm/install, and a
# generated blob (cd test_data && python3 generator.py).

set -euo pipefail

N="${1:?usage: check_layer.sh <N> [tiles]}"
TILES="${2:-4}"

ROOT=$(git rev-parse --show-toplevel)
TEST_DIR="$ROOT/tests/spatz_on_magia/onnx_mobilevit_v2"
BUILD="$ROOT/build"

export PATH="/home/calind/riscv/bin:$PATH"
export LLVM_INSTALL_DIR="${LLVM_INSTALL_DIR:-$ROOT/llvm/install}"

echo "=== configuring for MVIT_MAX_LAYERS=$N, tiles=$TILES ==="
# The cmake -D flag has to be seeded first, but the build itself must go through the
# Makefile: `make build tiles=N` is what regenerates
# targets/*/include/addr_map/tile_config.h, and that header is where NUM_HARTS comes
# from. Calling cmake alone leaves a stale NUM_HARTS baked into the binary while the
# simulator runs a different mesh size - the tiles past NUM_HARTS then do nothing.
cmake -DMVIT_MAX_LAYERS="$N" -S "$ROOT" -B "$BUILD" >/dev/null
make -C "$ROOT" build spatz=1 tiles="$TILES" test=onnx_mobilevit_v2 2>&1 | tail -1

grep -q "MESH_X_TILES $TILES" "$ROOT/targets/magia_v3/include/addr_map/tile_config.h" \
  || { echo "tile_config.h did not pick up tiles=$TILES" >&2; exit 1; }

echo "=== running ==="
OUT=$(PATH="$ROOT/gvsoc_venv/bin:$PATH" "$ROOT/gvsoc/install/bin/gvrun" \
        --target magia_v3 --work-dir "$ROOT/gvsoc_work" \
        --param binary="$BUILD/bin/onnx_mobilevit_v2" --trace-level=trace run \
        --attr magia_v3/n_tiles_x="$TILES" --attr magia_v3/n_tiles_y="$TILES" \
        --attr magia_v3/nb_pulp_cores=8 \
        --attr magia_v3/spatz_romfile="$BUILD/bin/bootrom/spatz_init.bin" \
        --trace=kill-module 2>&1)

echo "$OUT" | grep -E "stopped after|head:|tail:" || { echo "$OUT" | tail -20; exit 1; }

echo "=== expected ==="
cd "$TEST_DIR/test_data"
python3 - "$N" <<'PY'
import glob, os, sys
import numpy as np

n = int(sys.argv[1])
# Layer indices in the table are dense; the .npy files are named by ONNX node index, so
# take the n-th in sorted order.
files = sorted(glob.glob(os.path.join("layers", "*.npy")))
path = files[n - 1]
y = np.load(path).reshape(-1)
print(f"{os.path.basename(path)} ({y.size} elements)")
print("head:", " ".join(f"{v:04x}" for v in y[:8].view(np.uint16)))
print("tail:", " ".join(f"{v:04x}" for v in y[-8:].view(np.uint16)))
PY
