#  Copyright (C) 2023-2024 ETH Zurich and University of Bologna
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  SPDX-License-Identifier: Apache-2.0
# 
#  Authors: Victor Isachi <victor.isachi@unibo.it>
#  
#  Random uint16_t Z=Y+X*W FP16 array generator
#
#  Usage: python3 mat_gen.py <M_SIZE> <N_SIZE> <K_SIZE> <file_path>.c --seed <seed> --range <range>

import argparse
import numpy as np
from typing import List

def fmt_hex16_array(u16: np.ndarray, per_line: int=16) -> str:
  items = [f"0x{int(v):04X}" for v in u16.tolist()]
  lines = []
  for i in range(0, len(items), per_line):
    lines.append(" " + ", ".join(items[i:i+per_line]))
  return ",\n".join(lines)

def main():
  p = argparse.ArgumentParser(description="Generate C uint16_t arrays X, W, Y, Z with Z=Y+X*W.")
  p.add_argument("M_SIZE", type=int)
  p.add_argument("N_SIZE", type=int)
  p.add_argument("K_SIZE", type=int)
  p.add_argument("file_name", type=str, help="Output C file path/name")
  p.add_argument("--seed", type=int, default=1, help="Optional RNG seed")
  p.add_argument("--range", type=float, default=1.0, help="Optional range of values [-range, range]")
  args = p.parse_args()

  M, N, K = args.M_SIZE, args.N_SIZE, args.K_SIZE
  if any(d <= 0 for d in (M, N, K)):
    raise SystemExit("All dimensions must be positive integers.")

  rng = np.random.default_rng(args.seed)

  X = rng.uniform(-args.range, args.range, size=(M, N)).astype(np.float16)
  W = rng.uniform(-args.range, args.range, size=(N, K)).astype(np.float16)
  Y = rng.uniform(-args.range, args.range, size=(M, K)).astype(np.float16)

  Z = (Y+(X@W).astype(np.float16)).astype(np.float16)

  x_bits = X.view(np.uint16).ravel(order="C")
  w_bits = W.view(np.uint16).ravel(order="C")
  y_bits = Y.view(np.uint16).ravel(order="C")
  z_bits = Z.view(np.uint16).ravel(order="C")

  x_name = f"x_in_{M}x{N}"
  w_name = f"w_in_{N}x{K}"
  y_name = f"y_in_{M}x{K}"
  z_name = f"z_out_{M}x{K}"

  lines = []
  lines.append(f"// Auto-generated data (uint16_t) for Z = Y + X*W, with X {M}x{N}, W {N}x{K}, Y {M}x{K} and Z {M}x{K}")
  lines.append("// Python computes in full precision, values are truncated to 16 bits when emitted")
  lines.append(f"// RNG seed: {args.seed}\n")

  lines.append(f"#ifndef _MAT_GEN_{M}x{N}x{K}_")
  lines.append(f"#define _MAT_GEN_{M}x{N}x{K}_\n")

  lines.append(f"#define M_SIZE ({M})")
  lines.append(f"#define N_SIZE ({N})")
  lines.append(f"#define K_SIZE ({K})\n")

  lines.append(f"extern uint16_t {x_name} [M_SIZE*N_SIZE] = {{")
  lines.append(fmt_hex16_array(x_bits))
  lines.append("};\n")

  lines.append(f"extern uint16_t {w_name} [N_SIZE*K_SIZE] = {{")
  lines.append(fmt_hex16_array(w_bits))
  lines.append("};\n")

  lines.append(f"extern uint16_t {y_name} [M_SIZE*K_SIZE] = {{")
  lines.append(fmt_hex16_array(y_bits))
  lines.append("};\n")

  lines.append(f"extern uint16_t {z_name} [M_SIZE*K_SIZE] = {{")
  lines.append(fmt_hex16_array(z_bits))
  lines.append("};\n")

  lines.append(f"#endif /*_MAT_GEN_{M}x{N}x{K}_*/")

  with open(args.file_name, "w") as f:
    f.write("\n".join(lines))

  print(f"Wrote {args.file_name} with arrays:")
  print(f"{x_name}[{M}x{N}]")
  print(f"{w_name}[{N}x{K}]")
  print(f"{y_name}[{M}x{K}]")
  print(f"{z_name}[{M}x{K}]")

if __name__ == "__main__":
  main()