#!/usr/bin/env python3
# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""
gvsoc2perfetto.py — convert GVSOC waveform traces (VCD/FST) to Perfetto traces.

Usage:
    # FST: convert to VCD first (fst2vcd ships with GTKWave)
    fst2vcd all.fst -o all.vcd
    ./gvsoc2perfetto.py all.vcd [-o trace.perfetto-trace] [--pid-depth 2] [--include REGEX]

    Open the output at https://ui.perfetto.dev

Output format (--format):
  * perfetto (default) — native protobuf (Perfetto TrackEvent). Smaller and
        faster to load than JSON, and thread tracks render as just their label
        (e.g. "cv32-core.busy") with no sequential tid appended. Tile groups
        still show their instance number ("magia-v2-soc.magia-tile 14").
  * json               — Chrome Trace-Event JSON (the older format; Perfetto's
        UI appends a numeric tid to every thread name).
  --gzip writes a gzip-compressed file, which Perfetto opens directly.

Mapping:
  * string events  (vp::TraceEvent::event_string)  -> Perfetto slices (B/E).
        A new string opens a slice named after it; empty string / "idle"
        closes the current slice.
  * 1-bit events   (busy-style)                    -> slices named after the
        signal leaf while the value is 1.
  * multi-bit events                               -> Perfetto counter tracks,
        unless matched by --state-map (see below).
  * 'pc'/'active_pc' leaves                        -> hex-address slices
        (e.g. "0xc207b7b3") instead of a decimal counter track.

Track layout:
  * pid  = hierarchy prefix of depth --pid-depth   (e.g. chip.cluster_0)
  * tid  = one thread per signal                   (e.g. pe0.state)

Extra flags:
  --state-map 'LEAF=V:NAME,V:NAME,...'   (repeatable) Numeric signals whose
        leaf name matches LEAF are rendered as named B/E slices instead of a
        counter track, translating each value through the given table
        (unmapped values render as "state<N>"). E.g. for a redmule-style FSM:
        --state-map 'fsm_state=0:idle,1:preload,2:routine,3:storing,4:finished,5:acknowledge'
  --derive-busy 'LEAF_A,LEAF_B'           (repeatable) For every group of
        signals sharing an immediate parent path, synthesizes a
        '<parent>.busy_derived' 1-bit pseudo-signal active whenever
        value(LEAF_A) != value(LEAF_B) — a rough "is this block doing
        something" proxy when no real busy/state signal is instrumented.
  --rename 'OLD=NEW'                      (repeatable) Cosmetic-only: renames
        a hierarchy path component in displayed process/thread names (e.g.
        --rename 'ara=vfu' shows "snitch-spatz.vfu.label" instead of
        "snitch-spatz.ara.label"). Never affects --include/--exclude matching,
        which still sees the real VCD names.
  --split-asm                            Split each 'asm' string signal (format
        "<8 hex pc>_<disasm with '_' separators>_") into two tracks: a
        '<parent>.pc' track with slices named "0x<8 hex>" and a
        '<parent>.instruction' track with the disassembly (leading/trailing
        '_' removed, remaining '_' turned into spaces). Doubles the number of
        instruction-trace slices, since each 'asm' slice now produces one on
        each track.
  --stats                                 Also print per-track phase-duration
        totals to stdout (sum/count per pid/tid/slice-name), sorted by total
        duration — a quick sanity check without opening a browser.
"""

import argparse
import gzip
import itertools
import json
import re
import struct
import sys
from collections import defaultdict

# --------------------------------------------------------------------------
# Minimal VCD parser (vcdvcd drops string-type changes, so we roll our own).
# --------------------------------------------------------------------------

TIMESCALE_FACTORS = {"s": 1e12, "ms": 1e9, "us": 1e6, "ns": 1e3, "ps": 1.0, "fs": 1e-3}


def parse_vcd(path, include_re=None, exclude_re=None):
    """Return (signals, changes, ps_per_tick).

    signals: id -> dict(name=<dot path>, type=<'string'|'wire'|...>, width=int)
    changes: id -> list[(time_ticks, value_str)]

    If include_re/exclude_re are given, signals whose full dotted name doesn't
    match are dropped at definition time (not stored in `signals`), and their
    value changes are skipped as they're parsed — this avoids building
    per-signal change lists for the ~99% of a full-hierarchy dump that will
    just be filtered out later anyway (multi-GB VCDs otherwise parse slowly,
    see gvsoc-perfetto-notes.md).
    """
    signals = {}
    changes = defaultdict(list)
    scope = []
    ps_per_tick = 1.0
    time = 0
    in_defs = True

    def wanted(name):
        if include_re and not include_re.search(name):
            return False
        if exclude_re and exclude_re.search(name):
            return False
        return True

    with open(path, "r", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if in_defs:
                tok = line.split()
                if tok[0] == "$timescale":
                    # may be same-line ("$timescale 1ps $end") or multi-line
                    body = line
                    while "$end" not in body:
                        body += " " + next(f).strip()
                    m = re.search(r"(\d+)\s*(fs|ps|ns|us|ms|s)", body)
                    if m:
                        ps_per_tick = int(m.group(1)) * TIMESCALE_FACTORS[m.group(2)]
                elif tok[0] == "$scope":
                    scope.append(tok[2])
                elif tok[0] == "$upscope":
                    scope.pop()
                elif tok[0] == "$var":
                    # $var <type> <width> <id> <name...> $end
                    vtype, width, vid = tok[1], int(tok[2]), tok[3]
                    name = " ".join(tok[4:-1]) if tok[-1] == "$end" else " ".join(tok[4:])
                    name = re.sub(r"\s*\[.*\]$", "", name)  # strip [31:0]
                    full_name = ".".join(scope + [name])
                    if wanted(full_name):
                        signals[vid] = {
                            "name": full_name,
                            "type": vtype,
                            "width": width,
                        }
                elif tok[0] == "$enddefinitions":
                    in_defs = False
                continue

            # ---- value-change section ----
            c = line[0]
            if c == "#":
                time = int(line[1:])
            elif c in "01xXzZ":
                vid = line[1:]
                if vid in signals:
                    changes[vid].append((time, line[0].lower()))
            elif c in "bB":
                val, vid = line[1:].split()
                if vid in signals:
                    changes[vid].append((time, val))
            elif c in "rR":
                val, vid = line[1:].split()
                if vid in signals:
                    changes[vid].append((time, val))
            elif c in "sS":
                parts = line[1:].split()
                if len(parts) == 2:
                    val, vid = parts
                elif len(parts) == 1:          # "s <id>" -> empty string
                    val, vid = "", parts[0]
                else:                          # value contained spaces
                    val, vid = " ".join(parts[:-1]), parts[-1]
                if vid in signals:
                    changes[vid].append((time, val))
            elif c == "$":
                pass  # $dumpvars / $end etc.

    return signals, changes, ps_per_tick


# --------------------------------------------------------------------------
# --derive-busy: synthesize a busy-style pseudo-signal from a pair of
# sibling signals (e.g. idma's next_transfer_id/completed_id) when no real
# busy/state trace event exists for a component.
# --------------------------------------------------------------------------

def inject_derived_busy(signals, changes, specs):
    """For each (leaf_a, leaf_b) in specs, and each parent path that has both
    leaves among its children, add a synthetic 1-bit '<parent>.busy_derived'
    signal equal to (value(leaf_a) != value(leaf_b)) over their merged
    timeline. Mutates signals/changes in place.
    """
    by_parent = defaultdict(dict)
    for vid, sig in signals.items():
        parent, _, leaf = sig["name"].rpartition(".")
        by_parent[parent][leaf] = vid

    def as_num(val):
        try:
            return int(val, 2) if set(val) <= {"0", "1"} else int(val, 0)
        except ValueError:
            return None

    n_derived = 0
    for leaf_a, leaf_b in specs:
        for parent, leaves in by_parent.items():
            if leaf_a not in leaves or leaf_b not in leaves:
                continue
            vid_a, vid_b = leaves[leaf_a], leaves[leaf_b]
            merged = sorted(
                [(t, "a", v) for t, v in changes.get(vid_a, [])]
                + [(t, "b", v) for t, v in changes.get(vid_b, [])]
            )
            derived = []
            val_a = val_b = None
            for t, which, v in merged:
                if which == "a":
                    val_a = as_num(v)
                else:
                    val_b = as_num(v)
                if val_a is None or val_b is None:
                    continue
                derived.append((t, "1" if val_a != val_b else "0"))
            if not derived:
                continue
            new_vid = f"__derived__{parent}.{leaf_a}_ne_{leaf_b}"
            signals[new_vid] = {
                "name": f"{parent}.busy_derived({leaf_a}!={leaf_b})",
                "type": "wire",
                "width": 1,
            }
            changes[new_vid] = derived
            n_derived += 1
    return n_derived


# --------------------------------------------------------------------------
# --state-map: render specific numeric signals as named slices instead of
# counter tracks (e.g. light_redmule's fsm_state enum).
# --------------------------------------------------------------------------

def parse_state_map_arg(spec):
    """'fsm_state=0:idle,1:preload,...' -> (compiled_leaf_regex, {0: 'idle', ...})"""
    leaf_pattern, mapspec = spec.split("=", 1)
    mapping = {}
    for token in mapspec.split(","):
        v, name = token.split(":", 1)
        mapping[int(v, 0)] = name
    return re.compile(leaf_pattern), mapping


# --------------------------------------------------------------------------
# Perfetto (Chrome Trace Event JSON) emission
# --------------------------------------------------------------------------

IDLE_STRINGS = {"", "idle", "IDLE", "sleep", "off"}

# Program-counter-like leaves: rendered as hex-address slices (e.g.
# "0xc207b7b3") instead of the raw decimal counter track multi-bit signals
# otherwise get.
HEX_LEAVES = {"pc", "active_pc"}


def convert(signals, changes, ps_per_tick, pid_depth, include_re, exclude_re, state_maps=None,
            renames=None, split_asm=False):
    events = []
    pid_ids, tid_ids = {}, {}
    pid_used_nums = set()
    pid_child_prefix = {}  # pid -> "<word>-<num>-" prefix to strip from that pid's thread labels
    fallback_pid = itertools.count(10000)
    state_maps = state_maps or []
    renames = renames or {}

    def us(t):  # ticks -> microseconds (Perfetto JSON uses us)
        return t * ps_per_tick / 1e6

    def rename(label):
        # Cosmetic-only: swap an internal/codename hierarchy component (e.g.
        # "ara", the vector unit's actual RTL codename) for a more recognizable
        # one (e.g. "vfu") in displayed process/thread names. Never affects
        # --include/--exclude matching, which still sees the real VCD names.
        if not renames:
            return label
        return ".".join(renames.get(comp, comp) for comp in label.split("."))

    def get_pid(path_parts):
        # Perfetto's UI always displays process groups as "{name} {pid}". If the
        # group's own name already ends in an instance number (e.g. "magia-tile-0"),
        # showing an unrelated sequential pid on top of it reads as a confusing
        # double number ("magia-tile-0 3"). Instead, reuse that instance number as
        # the actual pid and strip it from the name, so Perfetto shows the single,
        # meaningful "magia-v2-soc.magia-tile 0".
        raw_key = ".".join(path_parts[:pid_depth]) or "top"
        if raw_key not in pid_ids:
            last = path_parts[min(pid_depth, len(path_parts)) - 1]
            m = re.search(r"-(\d+)$", last)
            pid_val = None
            inst = None            # the group's instance number (tile N), else None
            display_name = raw_key
            child_prefix = None
            if m:
                num = int(m.group(1))
                if num not in pid_used_nums:
                    pid_val = inst = num
                    display_name = raw_key[:-len(m.group(0))]
                    stripped_last = last[:-len(m.group(0))]
                    word = stripped_last.rsplit("-", 1)[-1] if stripped_last else None
                    if word:
                        # Child components repeat the instance number with just the
                        # last word (e.g. "tile-0-cv32-core"), not the full group
                        # name, so derive that shorter prefix separately.
                        child_prefix = f"{word}-{num}-"
            if pid_val is None:
                pid_val = next(fallback_pid)
            pid_used_nums.add(pid_val)
            pid_ids[raw_key] = pid_val
            pid_child_prefix[pid_val] = child_prefix
            # "num" lets the protobuf backend order groups (tiles by number) and
            # decide which groups show a number; None => a numberless group (NoC/L2).
            events.append({"ph": "M", "name": "process_name", "pid": pid_val,
                           "args": {"name": rename(display_name), "num": inst}})
        return pid_ids[raw_key]

    def get_tid(pid, label):
        # Same reasoning as get_pid: strip the redundant tile-instance prefix
        # (already implied by which process group this thread lives in) so
        # Perfetto's appended tid isn't sitting next to a duplicate number.
        prefix = pid_child_prefix.get(pid)
        if prefix:
            label = ".".join(
                comp[len(prefix):] if comp.startswith(prefix) else comp
                for comp in label.split(".")
            )
        label = rename(label)
        key = (pid, label)
        if key not in tid_ids:
            tid_ids[key] = len(tid_ids) + 1
            events.append({"ph": "M", "name": "thread_name", "pid": pid,
                           "tid": tid_ids[key], "args": {"name": label}})
        return tid_ids[key]

    from tqdm import tqdm
    for vid, sig in tqdm(signals.items()):
        name = sig["name"]
        if include_re and not include_re.search(name):
            continue
        if exclude_re and exclude_re.search(name):
            continue
        tv = changes.get(vid)
        if not tv:
            continue

        parts = name.split(".")
        pid = get_pid(parts)
        sub = ".".join(parts[pid_depth:]) or parts[-1]
        leaf = parts[-1]

        # --- --split-asm: one 'asm' string signal (e.g. "c207b7b3_p.bclri_x15_")
        # becomes two tracks: '<parent>.pc' with slices named "0x<8 hex>" and
        # '<parent>.instruction' with the disassembly (leading/trailing '_'
        # dropped, remaining '_' -> spaces). Both share the asm timing. ---
        if split_asm and sig["type"] == "string" and leaf == "asm":
            base = sub.rpartition(".")[0]
            tid_ins = get_tid(pid, f"{base}.instruction" if base else "instruction")
            open_pc = open_ins = False
            for t, val in tv:
                if open_ins:
                    events.append({"ph": "E", "pid": pid, "tid": tid_ins, "ts": us(t)})
                    open_ins = False
                if val not in IDLE_STRINGS:
                    instr = val[8:].strip("_").replace("_", " ")
                    if instr:
                        events.append({"ph": "B", "name": instr, "pid": pid,
                                       "tid": tid_ins, "ts": us(t)})
                        open_ins = True

            # If the last asm value is non-idle, ensure slices are closed.
            end_t = tv[-1][0] + 1
            if open_ins:
                events.append({"ph": "E", "pid": pid, "tid": tid_ins, "ts": us(end_t)})
            continue

        tid = get_tid(pid, sub)

        state_map = None
        for leaf_regex, mapping in state_maps:
            if leaf_regex.search(leaf):
                state_map = mapping
                break

        if leaf in HEX_LEAVES and sig["type"] != "string":
            nibbles = (sig["width"] + 3) // 4
            open_slice = False
            for t, val in tv:
                try:
                    num = int(val, 2) if set(val) <= {"0", "1"} else int(val, 0)
                except ValueError:
                    continue  # x/z
                if open_slice:
                    events.append({"ph": "E", "pid": pid, "tid": tid, "ts": us(t)})
                events.append({"ph": "B", "name": f"0x{num:0{nibbles}x}", "pid": pid,
                               "tid": tid, "ts": us(t)})
                open_slice = True

            if open_slice:
                events.append({"ph": "E", "pid": pid, "tid": tid, "ts": us(tv[-1][0] + 1)})
        elif state_map is not None:
            open_slice = False
            for t, val in tv:
                try:
                    num = int(val, 2) if set(val) <= {"0", "1"} else int(val, 0)
                except ValueError:
                    continue  # x/z states
                state_name = state_map.get(num, f"state{num}")
                if open_slice:
                    events.append({"ph": "E", "pid": pid, "tid": tid, "ts": us(t)})
                    open_slice = False
                if state_name not in IDLE_STRINGS:
                    events.append({"ph": "B", "name": state_name, "pid": pid, "tid": tid,
                                   "ts": us(t)})
                    open_slice = True

            if open_slice:
                events.append({"ph": "E", "pid": pid, "tid": tid, "ts": us(tv[-1][0] + 1)})
        elif sig["type"] == "string":
            open_slice = False
            for t, val in tv:
                if open_slice:
                    events.append({"ph": "E", "pid": pid, "tid": tid, "ts": us(t)})
                    open_slice = False
                if val not in IDLE_STRINGS:
                    events.append({"ph": "B", "name": val, "pid": pid, "tid": tid,
                                   "ts": us(t)})
                    open_slice = True

            if open_slice:
                events.append({"ph": "E", "pid": pid, "tid": tid, "ts": us(tv[-1][0] + 1)})
        elif sig["width"] == 1:
            open_slice = False
            for t, val in tv:
                if val == "1" and not open_slice:
                    events.append({"ph": "B", "name": leaf, "pid": pid, "tid": tid,
                                   "ts": us(t)})
                    open_slice = True
                elif val != "1" and open_slice:
                    events.append({"ph": "E", "pid": pid, "tid": tid, "ts": us(t)})
                    open_slice = False

            if open_slice:
                events.append({"ph": "E", "pid": pid, "tid": tid, "ts": us(tv[-1][0] + 1)})
        else:  # multi-bit -> counter track
            for t, val in tv:
                try:
                    num = int(val, 2) if set(val) <= {"0", "1"} else float(val)
                except ValueError:
                    continue  # x/z states
                events.append({"ph": "C", "name": leaf, "pid": pid, "tid": tid,
                               "ts": us(t), "args": {leaf: num}})

    return events


def print_stats(events):
    """Per (process, thread, slice-name) total duration / count, sorted desc."""
    proc_names, thread_names = {}, {}
    for e in events:
        if e["ph"] == "M":
            if e["name"] == "process_name":
                proc_names[e["pid"]] = e["args"]["name"]
            elif e["name"] == "thread_name":
                thread_names[(e["pid"], e["tid"])] = e["args"]["name"]

    open_b = {}
    totals = defaultdict(lambda: [0.0, 0])
    for e in events:
        key = (e.get("pid"), e.get("tid"))
        if e["ph"] == "B":
            open_b[key] = (e["name"], e["ts"])
        elif e["ph"] == "E" and key in open_b:
            name, ts0 = open_b.pop(key)
            entry = totals[(e["pid"], e["tid"], name)]
            entry[0] += e["ts"] - ts0
            entry[1] += 1

    rows = sorted(
        ((dur, proc_names.get(pid, str(pid)), thread_names.get((pid, tid), str(tid)), name, cnt)
         for (pid, tid, name), (dur, cnt) in totals.items()),
        reverse=True,
    )
    print(f"{'us total':>14}  {'count':>7}  process / thread / phase")
    for dur, proc, thread, name, cnt in rows[:200]:
        print(f"{dur:14.1f}  {cnt:7d}  {proc} / {thread} / {name}")
    if len(rows) > 200:
        print(f"... ({len(rows) - 200} more rows omitted)")


# --------------------------------------------------------------------------
# Perfetto native-protobuf emission (Trace = stream of TracePacket).
#
# We hand-roll the tiny, stable subset of the Perfetto proto we need rather
# than depending on the `protobuf`/`perfetto` packages or a .proto compile step
# (the VCD parser above is likewise dependency-free). The payoff over the JSON
# path: arbitrary track *names* (a generic named track renders as just "cv32-
# core.busy", with no Chrome-JSON "{name} {tid}" suffix), plus a smaller,
# faster-to-load binary (varint fields + interned repeated slice names).
# --------------------------------------------------------------------------

# --- protobuf wire primitives (wire types: 0=varint, 1=64-bit, 2=len-delim) ---

def _uvarint(n):
    n &= (1 << 64) - 1  # int64/uint64 share this encoding; negatives -> 10 bytes
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _tag(field, wire):
    return _uvarint((field << 3) | wire)


def _pv(field, n):          # varint field
    return _tag(field, 0) + _uvarint(n)


def _pd(field, x):          # double field
    return _tag(field, 1) + struct.pack("<d", x)


def _pb(field, b):          # length-delimited field (bytes / string / message)
    return _tag(field, 2) + _uvarint(len(b)) + b


def _ps(field, s):          # string field
    return _pb(field, s.encode("utf-8"))


# Perfetto enum/flag constants (see protobuf field table in the plan / docs).
_SEQ_CLEARED = 1   # TracePacket.SequenceFlags.SEQ_INCREMENTAL_STATE_CLEARED
_SEQ_NEEDS = 2     # TracePacket.SequenceFlags.SEQ_NEEDS_INCREMENTAL_STATE
_EV_BEGIN, _EV_END, _EV_COUNTER = 1, 2, 4  # TrackEvent.Type
_ORDER_EXPLICIT = 3  # TrackDescriptor.ChildTracksOrdering.EXPLICIT (order by sibling_order_rank)


def write_perfetto(events, path, intern=True, gzip_out=False):
    """Translate the JSON-event IR from convert() into a Perfetto protobuf.

    Track model: a single root track (child_ordering=EXPLICIT) groups one
    generic track per pid; tile groups are named "<short> <N>" and ranked by N
    (so they list 0..N in order), while numberless groups (NoC, L2) are named
    plainly and ranked after the tiles. Under each group is one generic named
    child track per (pid, tid), rendered as just the label (no sequential tid).
    Generic tracks are used (rather than ProcessDescriptor tracks) because
    Perfetto ignores ordering hints on process/thread tracks and always appends
    their pid to the name. Each child track gets its own packet sequence so its
    events stay timestamp-ordered within the sequence (Perfetto requires
    monotonic timestamps per sequence, and the IR emits each track's events
    contiguously in time order); slice-begin names are interned per sequence.
    """
    # Group the flat IR by track, preserving first-seen order.
    procs = {}          # pid -> {"name": display, "num": int or None}
    tracks = {}         # (pid, tid) -> {"label", "counter", "events": [(ts,kind,payload)]}
    order = []          # (pid, tid) in first-seen order
    for e in events:
        ph = e["ph"]
        if ph == "M":
            if e["name"] == "process_name":
                procs.setdefault(e["pid"], {"name": e["args"]["name"],
                                            "num": e["args"].get("num")})
            elif e["name"] == "thread_name":
                key = (e["pid"], e["tid"])
                if key not in tracks:
                    tracks[key] = {"label": e["args"]["name"], "counter": False, "events": []}
                    order.append(key)
            continue
        key = (e["pid"], e["tid"])
        tr = tracks.get(key)
        if tr is None:  # defensive: event before its thread_name metadata
            tr = tracks[key] = {"label": str(e["tid"]), "counter": False, "events": []}
            order.append(key)
        ts = round(e["ts"] * 1000)  # convert.us() gives microseconds; native clock is ns
        if ph == "B":
            tr["events"].append((ts, "B", e["name"]))
        elif ph == "E":
            tr["events"].append((ts, "E", None))
        elif ph == "C":
            tr["counter"] = True
            tr["events"].append((ts, "C", next(iter(e["args"].values()))))

    uid = itertools.count(1)
    root_uuid = next(uid)
    proc_uuid = {pid: next(uid) for pid in procs}
    track_uuid = {key: next(uid) for key in order}

    # Root name = the common leading dotted prefix of all group display names
    # (e.g. "magia-v2-soc"); it becomes the single top-level group and is
    # stripped from each child group's shown name.
    split_names = [p["name"].split(".") for p in procs.values()]
    common = []
    for comp in zip(*split_names):
        if len(set(comp)) == 1:
            common.append(comp[0])
        else:
            break
    root_name = ".".join(common) or "trace"
    strip = root_name + "." if common else ""

    def group_name_and_rank(info, unnumbered_idx):
        short = info["name"]
        if strip and short.startswith(strip):
            short = short[len(strip):]
        short = short or info["name"].rsplit(".", 1)[-1]
        if info["num"] is not None:                 # tile: keep number, order by it
            return f"{short} {info['num']}", info["num"]
        return short, 10 ** 6 + unnumbered_idx      # NoC/L2: no number, after tiles

    opener = gzip.open if gzip_out else open
    with opener(path, "wb") as f:
        def emit(packet):
            f.write(_pb(1, packet))  # Trace.packet (field 1, repeated)

        # --- structure sequence: declare every track up front ---
        SEQ_STRUCT = 1
        # Root: TrackDescriptor{uuid, name, child_ordering=EXPLICIT}
        emit(_pb(60, _pv(1, root_uuid) + _ps(2, root_name) + _pv(11, _ORDER_EXPLICIT))
             + _pv(10, SEQ_STRUCT))
        unnumbered = 0
        for pid, info in procs.items():
            name, rank = group_name_and_rank(info, unnumbered)
            if info["num"] is None:
                unnumbered += 1
            # Generic group TrackDescriptor{uuid, name, parent_uuid=root, sibling_order_rank}
            td = _pv(1, proc_uuid[pid]) + _ps(2, name) + _pv(5, root_uuid) + _pv(12, rank)
            emit(_pb(60, td) + _pv(10, SEQ_STRUCT))
        for key in order:
            tr = tracks[key]
            # TrackDescriptor{uuid, name, parent_uuid=group[, counter]}
            td = _pv(1, track_uuid[key]) + _ps(2, tr["label"]) + _pv(5, proc_uuid[key[0]])
            if tr["counter"]:
                td += _pb(8, b"")  # empty CounterDescriptor marks this a counter track
            emit(_pb(60, td) + _pv(10, SEQ_STRUCT))

        # --- one data sequence per track ---
        seq = itertools.count(2)
        for key in order:
            tr = tracks[key]
            evs = tr["events"]
            if not evs:
                continue
            tu = track_uuid[key]
            sid = next(seq)

            iid_of = {}
            if intern and not tr["counter"]:
                for _, kind, payload in evs:
                    if kind == "B" and payload not in iid_of:
                        iid_of[payload] = len(iid_of) + 1
                if iid_of:
                    interned = b"".join(
                        _pb(2, _pv(1, i) + _ps(2, n)) for n, i in iid_of.items()
                    )  # InternedData.event_names (field 2) -> EventName{iid, name}
                    emit(_pb(12, interned) + _pv(10, sid) + _pv(13, _SEQ_CLEARED))

            for ts, kind, payload in evs:
                if kind == "B":
                    if iid_of:
                        te = _pv(9, _EV_BEGIN) + _pv(11, tu) + _pv(10, iid_of[payload])
                        pkt = _pv(8, ts) + _pb(11, te) + _pv(10, sid) + _pv(13, _SEQ_NEEDS)
                    else:
                        te = _pv(9, _EV_BEGIN) + _pv(11, tu) + _ps(23, payload)
                        pkt = _pv(8, ts) + _pb(11, te) + _pv(10, sid)
                elif kind == "E":
                    te = _pv(9, _EV_END) + _pv(11, tu)
                    pkt = _pv(8, ts) + _pb(11, te) + _pv(10, sid)
                else:  # counter
                    cv = _pd(44, payload) if isinstance(payload, float) else _pv(30, int(payload))
                    te = _pv(9, _EV_COUNTER) + _pv(11, tu) + cv
                    pkt = _pv(8, ts) + _pb(11, te) + _pv(10, sid)
                emit(pkt)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vcd", help="input VCD (use fst2vcd for FST dumps)")
    ap.add_argument("-o", "--output", default=None,
                    help="output file (default: trace.perfetto-trace, or "
                         "trace.json with --format json)")
    ap.add_argument("--format", choices=["perfetto", "json"], default="perfetto",
                    help="output format: 'perfetto' native protobuf (default, smaller "
                         "and drops the sequential tid from thread names) or 'json' "
                         "Chrome Trace-Event")
    ap.add_argument("--no-intern", action="store_true",
                    help="perfetto: emit inline slice names instead of interning them "
                         "(larger file; useful for debugging)")
    ap.add_argument("--gzip", action="store_true",
                    help="gzip the output (Perfetto opens .gz directly)")
    ap.add_argument("--pid-depth", type=int, default=2,
                    help="hierarchy depth grouped as one Perfetto process (default 2)")
    ap.add_argument("--include", help="regex: only convert matching signal paths")
    ap.add_argument("--exclude", help="regex: skip matching signal paths")
    ap.add_argument("--state-map", action="append", default=[], metavar="LEAF=V:NAME,...",
                    help="render a numeric signal as named slices (repeatable)")
    ap.add_argument("--derive-busy", action="append", default=[], metavar="LEAF_A,LEAF_B",
                    help="synthesize a busy slice = LEAF_A != LEAF_B (repeatable)")
    ap.add_argument("--rename", action="append", default=[], metavar="OLD=NEW",
                    help="cosmetic-only: rename a hierarchy path component in "
                         "displayed process/thread names (repeatable)")
    ap.add_argument("--split-asm", action="store_true",
                    help="extracts a '.instruction' track (disassembly) from 'asm' string signal")
    ap.add_argument("--stats", action="store_true",
                    help="also print per-track phase-duration totals to stdout")
    args = ap.parse_args()

    inc = re.compile(args.include) if args.include else None
    exc = re.compile(args.exclude) if args.exclude else None
    signals, changes, ps = parse_vcd(args.vcd, inc, exc)

    if args.derive_busy:
        specs = []
        for s in args.derive_busy:
            parts = s.split(",", 1)
            if len(parts) != 2 or not parts[0] or not parts[1]:
                ap.error(f"--derive-busy expects 'LEAF_A,LEAF_B' (got {s!r})")
            specs.append((parts[0], parts[1]))
        n_derived = inject_derived_busy(signals, changes, specs)
        print(f"--derive-busy: synthesized {n_derived} busy_derived signal(s)", file=sys.stderr)

    state_maps = [parse_state_map_arg(s) for s in args.state_map]
    renames = dict(s.split("=", 1) for s in args.rename)

    # Filtering already happened in parse_vcd (and derived signals are
    # synthetic, so they wouldn't match the original --include anyway) —
    # convert() just emits everything it was handed.
    events = convert(signals, changes, ps, args.pid_depth, None, None, state_maps, renames,
                     args.split_asm)

    out = args.output or ("trace.perfetto-trace" if args.format == "perfetto" else "trace.json")
    if args.gzip and not out.endswith(".gz"):
        out += ".gz"

    if args.format == "perfetto":
        write_perfetto(events, out, intern=not args.no_intern, gzip_out=args.gzip)
    elif args.gzip:
        with gzip.open(out, "wt") as f:
            json.dump({"traceEvents": events, "displayTimeUnit": "ns"}, f)
    else:
        with open(out, "w") as f:
            json.dump({"traceEvents": events, "displayTimeUnit": "ns"}, f)

    n_tracks = len({(e.get("pid"), e.get("tid")) for e in events if e["ph"] != "M"})
    print(f"{out}: {len(events)} events on {n_tracks} tracks "
          f"({len(signals)} signals in VCD). Open at https://ui.perfetto.dev",
          file=sys.stderr)

    if args.stats:
        print_stats(events)


if __name__ == "__main__":
    main()
