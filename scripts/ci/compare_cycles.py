#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0, see LICENSE.APACHE for details.
# SPDX-License-Identifier: Apache-2.0

"""Cross-check per-tile cycle counts between an RTL (Modelsim) and a GVSoC log.

The same test binary is run on both simulators. Each tile emits one cycle count
for a single profiled region:

  * GVSoC -- the test prints, via the xperf_end() helper:
        [XPERF] mhartid <id> CYCLES <n>
  * RTL   -- the MAGIA testbench (magia_vip.sv, `ifdef PROFILE_SENTINEL) prints:
        [TB][mhartid <id> - Tile (y, x)] Detected sentinel start-end pair with
        latency <t>ns (<n> clock cycles)

For every tile (mhartid) present in both logs the relative difference must stay
within the tolerance, otherwise the script exits non-zero (hard CI failure).
"""

import argparse
import re
import sys

RTL_RE = re.compile(r"mhartid\s+(\d+).*?\((\d+)\s+clock cycles\)")
GVSOC_RE = re.compile(r"\[XPERF\]\s+mhartid\s+(\d+)\s+CYCLES\s+(\d+)")


def parse(path, regex):
    """Return {mhartid: cycles}. Keeps the last value seen for a given tile."""
    result = {}
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                m = regex.search(line)
                if m:
                    result[int(m.group(1))] = int(m.group(2))
    except FileNotFoundError:
        print(f"ERROR: log file not found: {path}")
        return None
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtl", required=True, help="RTL (Modelsim) log file")
    ap.add_argument("--gvsoc", required=True, help="GVSoC log file")
    ap.add_argument("--tol", type=float, default=0.10, help="relative tolerance")
    ap.add_argument("--test", default="<test>", help="test name (for reporting)")
    args = ap.parse_args()

    rtl = parse(args.rtl, RTL_RE)
    gvsoc = parse(args.gvsoc, GVSOC_RE)

    print(f"=== {args.test}: RTL vs GVSoC cycle cross-check (tol {args.tol:.0%}) ===")

    if rtl is None or gvsoc is None:
        return 1
    if not rtl:
        print(f"ERROR: no RTL cycle counts parsed from {args.rtl}")
        return 1
    if not gvsoc:
        print(f"ERROR: no GVSoC cycle counts parsed from {args.gvsoc}")
        return 1

    tiles = sorted(set(rtl) | set(gvsoc))
    failed = False
    print(f"{'mhartid':>8} {'rtl':>12} {'gvsoc':>12} {'delta%':>9}  result")
    for t in tiles:
        r = rtl.get(t)
        g = gvsoc.get(t)
        if r is None or g is None:
            failed = True
            missing = "RTL" if r is None else "GVSoC"
            print(f"{t:>8} {str(r):>12} {str(g):>12} {'-':>9}  FAIL (missing on {missing})")
            continue
        delta = abs(r - g) / max(r, g, 1)
        ok = delta <= args.tol
        failed = failed or not ok
        print(f"{t:>8} {r:>12} {g:>12} {delta*100:>8.2f}%  {'PASS' if ok else 'FAIL'}")

    if failed:
        print(f"RESULT: {args.test} FAILED")
        return 1
    print(f"RESULT: {args.test} PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
