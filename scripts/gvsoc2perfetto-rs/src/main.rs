// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// gvsoc2perfetto — convert GVSOC waveform traces (VCD) to Perfetto traces.
//

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufReader, BufRead, BufWriter, Write};

use flate2::write::GzEncoder;
use flate2::Compression;
use indexmap::IndexMap;
use indicatif::{ProgressBar, ProgressStyle};
use regex::Regex;

// --------------------------------------------------------------------------
// Data model
// --------------------------------------------------------------------------

struct Signal {
    name: String,   // full dotted path
    vtype: String,  // "wire" | "string" | "real" | ...
    width: i64,
}

struct Change {
    t: i64,       // time in ticks
    val: String,  // value token (lowercased scalar char, binary string, or string value)
}

#[derive(Clone)]
enum Num {
    I(u128),
    F(f64),
}

// Intermediate event IR, mirroring the JSON dicts convert() builds in Python.
enum Event {
    ProcName { pid: i64, name: String, num: Option<i64> },
    ThreadName { pid: i64, tid: i64, name: String },
    Begin { pid: i64, tid: i64, ts: f64, name: String },
    End { pid: i64, tid: i64, ts: f64 },
    Counter { pid: i64, tid: i64, ts: f64, name: String, val: Num },
}

const IDLE: [&str; 5] = ["", "idle", "IDLE", "sleep", "off"];

fn is_idle(s: &str) -> bool {
    IDLE.contains(&s)
}

fn is_hex_leaf(s: &str) -> bool {
    s == "pc" || s == "active_pc"
}

// --------------------------------------------------------------------------
// Numeric parsing helpers (mirror Python int(v,2) / int(v,0) / float(v))
// --------------------------------------------------------------------------

fn is_bits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b == b'0' || b == b'1')
}

// Python int(val, 0): auto-detect base from 0x/0b/0o prefix, else decimal.
fn parse_int0(s: &str) -> Option<u128> {
    let s = s.trim();
    let low = s.to_ascii_lowercase();
    if let Some(r) = low.strip_prefix("0x") {
        u128::from_str_radix(r, 16).ok()
    } else if let Some(r) = low.strip_prefix("0b") {
        u128::from_str_radix(r, 2).ok()
    } else if let Some(r) = low.strip_prefix("0o") {
        u128::from_str_radix(r, 8).ok()
    } else {
        s.parse::<u128>().ok()
    }
}

// Python: int(val, 2) if set(val) <= {"0","1"} else int(val, 0)
fn parse_bits_or_int0(s: &str) -> Option<u128> {
    if is_bits(s) {
        u128::from_str_radix(s, 2).ok()
    } else {
        parse_int0(s)
    }
}

fn atoi(b: &[u8]) -> i64 {
    let mut n = 0i64;
    for &d in b {
        if d.is_ascii_digit() {
            n = n * 10 + (d - b'0') as i64;
        }
    }
    n
}

// --------------------------------------------------------------------------
// VCD parser
// --------------------------------------------------------------------------

fn timescale_factor(u: &str) -> f64 {
    match u {
        "s" => 1e12,
        "ms" => 1e9,
        "us" => 1e6,
        "ns" => 1e3,
        "ps" => 1.0,
        "fs" => 1e-3,
        _ => 1.0,
    }
}

fn parse_vcd(
    path: &str,
    inc: Option<&Regex>,
    exc: Option<&Regex>,
) -> io::Result<(IndexMap<String, Signal>, HashMap<String, Vec<Change>>, f64)> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(1 << 20, file);

    let mut signals: IndexMap<String, Signal> = IndexMap::new();
    let mut changes: HashMap<String, Vec<Change>> = HashMap::new();
    let mut scope: Vec<String> = Vec::new();
    let mut ps_per_tick = 1.0f64;
    let mut time = 0i64;
    let mut in_defs = true;

    let bracket_re = Regex::new(r"\s*\[.*\]$").unwrap();
    let ts_re = Regex::new(r"(\d+)\s*(fs|ps|ns|us|ms|s)").unwrap();

    let wanted = |full: &str| -> bool {
        inc.map_or(true, |r| r.is_match(full)) && exc.map_or(true, |r| !r.is_match(full))
    };

    let mut buf: Vec<u8> = Vec::new();
    loop {
        buf.clear();
        let n = reader.read_until(b'\n', &mut buf)?;
        if n == 0 {
            break;
        }

        if in_defs {
            let line = String::from_utf8_lossy(&buf);
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let tok: Vec<&str> = line.split_whitespace().collect();
            match tok[0] {
                "$timescale" => {
                    // may be same-line ("$timescale 1ps $end") or multi-line
                    let mut body = line.to_string();
                    let mut tb: Vec<u8> = Vec::new();
                    while !body.contains("$end") {
                        tb.clear();
                        let m = reader.read_until(b'\n', &mut tb)?;
                        if m == 0 {
                            break;
                        }
                        body.push(' ');
                        body.push_str(String::from_utf8_lossy(&tb).trim());
                    }
                    if let Some(cap) = ts_re.captures(&body) {
                        let d: f64 = cap[1].parse().unwrap_or(1.0);
                        ps_per_tick = d * timescale_factor(&cap[2]);
                    }
                }
                "$scope" => {
                    if tok.len() >= 3 {
                        scope.push(tok[2].to_string());
                    }
                }
                "$upscope" => {
                    scope.pop();
                }
                "$var" => {
                    // $var <type> <width> <id> <name...> $end
                    if tok.len() >= 4 {
                        let vtype = tok[1];
                        let width: i64 = tok[2].parse().unwrap_or(0);
                        let vid = tok[3];
                        let name = if *tok.last().unwrap() == "$end" {
                            tok[4..tok.len() - 1].join(" ")
                        } else {
                            tok[4..].join(" ")
                        };
                        let name = bracket_re.replace(&name, "").into_owned(); // strip [31:0]
                        let full = if scope.is_empty() {
                            name.clone()
                        } else {
                            format!("{}.{}", scope.join("."), name)
                        };
                        if wanted(&full) {
                            signals.insert(
                                vid.to_string(),
                                Signal { name: full, vtype: vtype.to_string(), width },
                            );
                            changes.insert(vid.to_string(), Vec::new());
                        }
                    }
                }
                "$enddefinitions" => {
                    in_defs = false;
                }
                _ => {}
            }
            continue;
        }

        // ---- value-change section ----
        let lt = buf.trim_ascii();
        if lt.is_empty() {
            continue;
        }
        let c = lt[0];
        match c {
            b'#' => {
                time = atoi(&lt[1..]);
            }
            b'0' | b'1' | b'x' | b'X' | b'z' | b'Z' => {
                let vid = String::from_utf8_lossy(&lt[1..]);
                if let Some(v) = changes.get_mut(vid.as_ref()) {
                    v.push(Change { t: time, val: (c as char).to_ascii_lowercase().to_string() });
                }
            }
            b'b' | b'B' | b'r' | b'R' => {
                let s = String::from_utf8_lossy(&lt[1..]);
                let mut it = s.split_whitespace();
                let val = it.next().unwrap_or("");
                let vid = it.next().unwrap_or("");
                if !vid.is_empty() {
                    if let Some(v) = changes.get_mut(vid) {
                        v.push(Change { t: time, val: val.to_string() });
                    }
                }
            }
            b's' | b'S' => {
                let s = String::from_utf8_lossy(&lt[1..]);
                let parts: Vec<&str> = s.split_whitespace().collect();
                let (val, vid): (String, &str) = match parts.len() {
                    0 => continue,
                    1 => (String::new(), parts[0]), // "s <id>" -> empty string
                    2 => (parts[0].to_string(), parts[1]),
                    _ => (parts[..parts.len() - 1].join(" "), parts[parts.len() - 1]),
                };
                if let Some(v) = changes.get_mut(vid) {
                    v.push(Change { t: time, val });
                }
            }
            b'$' => {} // $dumpvars / $end etc.
            _ => {}
        }
    }

    Ok((signals, changes, ps_per_tick))
}

// --------------------------------------------------------------------------
// --derive-busy: synthesize a busy pseudo-signal from a pair of siblings.
// --------------------------------------------------------------------------

fn inject_derived_busy(
    signals: &mut IndexMap<String, Signal>,
    changes: &mut HashMap<String, Vec<Change>>,
    specs: &[(String, String)],
) -> usize {
    // parent path -> {leaf -> vid}, in signals insertion order
    let mut by_parent: IndexMap<String, IndexMap<String, String>> = IndexMap::new();
    for (vid, sig) in signals.iter() {
        let (parent, leaf) = match sig.name.rsplit_once('.') {
            Some((p, l)) => (p.to_string(), l.to_string()),
            None => (String::new(), sig.name.clone()),
        };
        by_parent.entry(parent).or_default().insert(leaf, vid.clone());
    }

    let mut new_signals: Vec<(String, Signal, Vec<Change>)> = Vec::new();
    for (leaf_a, leaf_b) in specs {
        for (parent, leaves) in by_parent.iter() {
            let (vid_a, vid_b) = match (leaves.get(leaf_a), leaves.get(leaf_b)) {
                (Some(a), Some(b)) => (a, b),
                _ => continue,
            };
            let mut merged: Vec<(i64, u8, &str)> = Vec::new();
            if let Some(cs) = changes.get(vid_a) {
                for c in cs {
                    merged.push((c.t, 0, c.val.as_str()));
                }
            }
            if let Some(cs) = changes.get(vid_b) {
                for c in cs {
                    merged.push((c.t, 1, c.val.as_str()));
                }
            }
            merged.sort_by(|x, y| x.0.cmp(&y.0).then(x.1.cmp(&y.1)).then(x.2.cmp(y.2)));

            let mut derived: Vec<Change> = Vec::new();
            let mut val_a: Option<u128> = None;
            let mut val_b: Option<u128> = None;
            for (t, which, v) in merged {
                let nv = parse_bits_or_int0(v);
                if which == 0 {
                    val_a = nv;
                } else {
                    val_b = nv;
                }
                if val_a.is_none() || val_b.is_none() {
                    continue;
                }
                derived.push(Change {
                    t,
                    val: if val_a != val_b { "1".to_string() } else { "0".to_string() },
                });
            }
            if derived.is_empty() {
                continue;
            }
            let new_vid = format!("__derived__{}.{}_ne_{}", parent, leaf_a, leaf_b);
            let name = format!("{}.busy_derived({}!={})", parent, leaf_a, leaf_b);
            new_signals.push((new_vid, Signal { name, vtype: "wire".to_string(), width: 1 }, derived));
        }
    }

    let n = new_signals.len();
    for (vid, sig, ch) in new_signals {
        signals.insert(vid.clone(), sig);
        changes.insert(vid, ch);
    }
    n
}

// --------------------------------------------------------------------------
// --state-map parsing
// --------------------------------------------------------------------------

fn parse_state_map_arg(spec: &str) -> (Regex, HashMap<u128, String>) {
    let (leaf, mapspec) = spec.split_once('=').expect("--state-map expects 'LEAF=V:NAME,...'");
    let mut m = HashMap::new();
    for token in mapspec.split(',') {
        let (v, name) = token.split_once(':').expect("--state-map token expects 'V:NAME'");
        let key = parse_int0(v).expect("--state-map value must be an integer");
        m.insert(key, name.to_string());
    }
    (Regex::new(leaf).expect("invalid --state-map leaf regex"), m)
}

// --------------------------------------------------------------------------
// convert(): signals + changes -> flat event IR
// --------------------------------------------------------------------------

struct Converter<'a> {
    events: Vec<Event>,
    pid_ids: HashMap<String, i64>,
    pid_used_nums: HashSet<i64>,
    pid_child_prefix: HashMap<i64, Option<String>>,
    tid_ids: HashMap<(i64, String), i64>,
    fallback_pid: i64,
    pid_depth: usize,
    renames: &'a HashMap<String, String>,
    last_num_re: Regex,
}

impl<'a> Converter<'a> {
    fn new(pid_depth: usize, renames: &'a HashMap<String, String>) -> Self {
        Converter {
            events: Vec::new(),
            pid_ids: HashMap::new(),
            pid_used_nums: HashSet::new(),
            pid_child_prefix: HashMap::new(),
            tid_ids: HashMap::new(),
            fallback_pid: 10000,
            pid_depth,
            renames,
            last_num_re: Regex::new(r"-(\d+)$").unwrap(),
        }
    }

    fn rename(&self, label: &str) -> String {
        if self.renames.is_empty() {
            return label.to_string();
        }
        label
            .split('.')
            .map(|c| self.renames.get(c).map(|s| s.as_str()).unwrap_or(c))
            .collect::<Vec<_>>()
            .join(".")
    }

    fn get_pid(&mut self, parts: &[&str]) -> i64 {
        let end = self.pid_depth.min(parts.len());
        let raw_key = {
            let s = parts[..end].join(".");
            if s.is_empty() {
                "top".to_string()
            } else {
                s
            }
        };
        if let Some(p) = self.pid_ids.get(&raw_key) {
            return *p;
        }

        let idx = end.saturating_sub(1); // min(pid_depth, len) - 1
        let last = parts[idx];
        let mut pid_val: Option<i64> = None;
        let mut inst: Option<i64> = None;
        let mut display_name = raw_key.clone();
        let mut child_prefix: Option<String> = None;

        if let Some(cap) = self.last_num_re.captures(last) {
            let num: i64 = cap[1].parse().unwrap();
            if !self.pid_used_nums.contains(&num) {
                pid_val = Some(num);
                inst = Some(num);
                let full = cap.get(0).unwrap().as_str(); // e.g. "-14"
                display_name = raw_key[..raw_key.len() - full.len()].to_string();
                let stripped_last = &last[..last.len() - full.len()];
                if !stripped_last.is_empty() {
                    // child components repeat the number with just the last word
                    let word = stripped_last.rsplit('-').next().unwrap();
                    if !word.is_empty() {
                        child_prefix = Some(format!("{}-{}-", word, num));
                    }
                }
            }
        }

        let pid_val = pid_val.unwrap_or_else(|| {
            let p = self.fallback_pid;
            self.fallback_pid += 1;
            p
        });
        self.pid_used_nums.insert(pid_val);
        self.pid_ids.insert(raw_key, pid_val);
        self.pid_child_prefix.insert(pid_val, child_prefix);
        let dn = self.rename(&display_name);
        self.events.push(Event::ProcName { pid: pid_val, name: dn, num: inst });
        pid_val
    }

    fn get_tid(&mut self, pid: i64, label: &str) -> i64 {
        let prefix = self.pid_child_prefix.get(&pid).cloned().flatten();
        let label2 = if let Some(pfx) = &prefix {
            label
                .split('.')
                .map(|c| if c.starts_with(pfx.as_str()) { &c[pfx.len()..] } else { c })
                .collect::<Vec<_>>()
                .join(".")
        } else {
            label.to_string()
        };
        let label2 = self.rename(&label2);
        let key = (pid, label2.clone());
        if let Some(t) = self.tid_ids.get(&key) {
            return *t;
        }
        let tid = self.tid_ids.len() as i64 + 1;
        self.tid_ids.insert(key, tid);
        self.events.push(Event::ThreadName { pid, tid, name: label2 });
        tid
    }
}

fn convert(
    signals: &IndexMap<String, Signal>,
    changes: &HashMap<String, Vec<Change>>,
    ps: f64,
    pid_depth: usize,
    state_maps: &[(Regex, HashMap<u128, String>)],
    renames: &HashMap<String, String>,
    split_asm: bool,
) -> Vec<Event> {
    let mut conv = Converter::new(pid_depth, renames);
    let us = |t: i64| -> f64 { t as f64 * ps / 1e6 };

    for (vid, sig) in signals.iter() {
        let tv = match changes.get(vid) {
            Some(v) if !v.is_empty() => v,
            _ => continue,
        };
        let parts: Vec<&str> = sig.name.split('.').collect();
        let pid = conv.get_pid(&parts);
        let sub = {
            let j = if pid_depth <= parts.len() { parts[pid_depth..].join(".") } else { String::new() };
            if j.is_empty() {
                parts[parts.len() - 1].to_string()
            } else {
                j
            }
        };
        let leaf = parts[parts.len() - 1];

        // --split-asm: 'asm' string signal -> '<parent>.instruction' track.
        if split_asm && sig.vtype == "string" && leaf == "asm" {
            let base = sub.rsplit_once('.').map(|x| x.0).unwrap_or("");
            let ins_label =
                if !base.is_empty() { format!("{}.instruction", base) } else { "instruction".to_string() };
            let tid_ins = conv.get_tid(pid, &ins_label);
            let mut open = false;
            for ch in tv {
                let ts = us(ch.t);
                if open {
                    conv.events.push(Event::End { pid, tid: tid_ins, ts });
                    open = false;
                }
                if !is_idle(&ch.val) {
                    let tail = ch.val.get(8..).unwrap_or("");
                    let instr = tail.trim_matches('_').replace('_', " ");
                    if !instr.is_empty() {
                        conv.events.push(Event::Begin { pid, tid: tid_ins, ts, name: instr });
                        open = true;
                    }
                }
            }
            if open {
                let ts = us(tv.last().unwrap().t + 1);
                conv.events.push(Event::End { pid, tid: tid_ins, ts });
            }
            continue;
        }

        let tid = conv.get_tid(pid, &sub);
        let state_map = state_maps.iter().find(|(re, _)| re.is_match(leaf)).map(|(_, m)| m);

        if is_hex_leaf(leaf) && sig.vtype != "string" {
            let nibbles = ((sig.width + 3) / 4) as usize;
            let mut open = false;
            for ch in tv {
                let num = match parse_bits_or_int0(&ch.val) {
                    Some(n) => n,
                    None => continue,
                };
                let ts = us(ch.t);
                if open {
                    conv.events.push(Event::End { pid, tid, ts });
                }
                conv.events.push(Event::Begin {
                    pid,
                    tid,
                    ts,
                    name: format!("0x{:0width$x}", num, width = nibbles),
                });
                open = true;
            }
            if open {
                let ts = us(tv.last().unwrap().t + 1);
                conv.events.push(Event::End { pid, tid, ts });
            }
        } else if let Some(map) = state_map {
            let mut open = false;
            for ch in tv {
                let num = match parse_bits_or_int0(&ch.val) {
                    Some(n) => n,
                    None => continue,
                };
                let ts = us(ch.t);
                let state_name = map.get(&num).cloned().unwrap_or_else(|| format!("state{}", num));
                if open {
                    conv.events.push(Event::End { pid, tid, ts });
                    open = false;
                }
                if !is_idle(&state_name) {
                    conv.events.push(Event::Begin { pid, tid, ts, name: state_name });
                    open = true;
                }
            }
            if open {
                let ts = us(tv.last().unwrap().t + 1);
                conv.events.push(Event::End { pid, tid, ts });
            }
        } else if sig.vtype == "string" {
            let mut open = false;
            for ch in tv {
                let ts = us(ch.t);
                if open {
                    conv.events.push(Event::End { pid, tid, ts });
                    open = false;
                }
                if !is_idle(&ch.val) {
                    conv.events.push(Event::Begin { pid, tid, ts, name: ch.val.clone() });
                    open = true;
                }
            }
            if open {
                let ts = us(tv.last().unwrap().t + 1);
                conv.events.push(Event::End { pid, tid, ts });
            }
        } else if sig.width == 1 {
            let mut open = false;
            for ch in tv {
                let ts = us(ch.t);
                if ch.val == "1" && !open {
                    conv.events.push(Event::Begin { pid, tid, ts, name: leaf.to_string() });
                    open = true;
                } else if ch.val != "1" && open {
                    conv.events.push(Event::End { pid, tid, ts });
                    open = false;
                }
            }
            if open {
                let ts = us(tv.last().unwrap().t + 1);
                conv.events.push(Event::End { pid, tid, ts });
            }
        } else {
            // multi-bit -> counter track
            for ch in tv {
                let ts = us(ch.t);
                let num = if is_bits(&ch.val) {
                    match u128::from_str_radix(&ch.val, 2) {
                        Ok(n) => Num::I(n),
                        Err(_) => continue,
                    }
                } else {
                    match ch.val.parse::<f64>() {
                        Ok(x) => Num::F(x),
                        Err(_) => continue,
                    }
                };
                conv.events.push(Event::Counter { pid, tid, ts, name: leaf.to_string(), val: num });
            }
        }
    }

    conv.events
}

// --------------------------------------------------------------------------
// Perfetto protobuf emission (hand-rolled wire encoding, mirrors the Python)
// --------------------------------------------------------------------------

fn uvarint(out: &mut Vec<u8>, mut n: u64) {
    loop {
        let b = (n & 0x7f) as u8;
        n >>= 7;
        if n != 0 {
            out.push(b | 0x80);
        } else {
            out.push(b);
            break;
        }
    }
}

fn tag(out: &mut Vec<u8>, field: u64, wire: u8) {
    uvarint(out, (field << 3) | wire as u64);
}

fn pv(out: &mut Vec<u8>, field: u64, n: u64) {
    tag(out, field, 0);
    uvarint(out, n);
}

fn pd(out: &mut Vec<u8>, field: u64, x: f64) {
    tag(out, field, 1);
    out.extend_from_slice(&x.to_le_bytes());
}

fn pb(out: &mut Vec<u8>, field: u64, body: &[u8]) {
    tag(out, field, 2);
    uvarint(out, body.len() as u64);
    out.extend_from_slice(body);
}

fn ps(out: &mut Vec<u8>, field: u64, s: &str) {
    pb(out, field, s.as_bytes());
}

fn emit(out: &mut Vec<u8>, packet: &[u8]) {
    pb(out, 1, packet); // Trace.packet (field 1, repeated)
}

// Perfetto enum/flag constants.
const SEQ_CLEARED: u64 = 1;
const SEQ_NEEDS: u64 = 2;
const EV_BEGIN: u64 = 1;
const EV_END: u64 = 2;
const EV_COUNTER: u64 = 4;
const ORDER_EXPLICIT: u64 = 3;
const SEQ_STRUCT: u64 = 1;

enum Ek {
    B(String),
    E,
    C(Num),
}

struct TrackData {
    label: String,
    counter: bool,
    events: Vec<(i64, Ek)>,
}

fn round_ns(ts_us: f64) -> i64 {
    (ts_us * 1000.0).round_ties_even() as i64
}

fn write_perfetto(events: &[Event], path: &str, intern: bool, gzip_out: bool) -> io::Result<()> {
    // Group the flat IR by track, preserving first-seen order.
    let mut procs: IndexMap<i64, (String, Option<i64>)> = IndexMap::new();
    let mut tracks: IndexMap<(i64, i64), TrackData> = IndexMap::new();

    for e in events {
        match e {
            Event::ProcName { pid, name, num } => {
                procs.entry(*pid).or_insert_with(|| (name.clone(), *num));
            }
            Event::ThreadName { pid, tid, name } => {
                tracks.entry((*pid, *tid)).or_insert_with(|| TrackData {
                    label: name.clone(),
                    counter: false,
                    events: Vec::new(),
                });
            }
            Event::Begin { pid, tid, ts, name } => {
                let td = tracks.entry((*pid, *tid)).or_insert_with(|| TrackData {
                    label: tid.to_string(),
                    counter: false,
                    events: Vec::new(),
                });
                td.events.push((round_ns(*ts), Ek::B(name.clone())));
            }
            Event::End { pid, tid, ts } => {
                let td = tracks.entry((*pid, *tid)).or_insert_with(|| TrackData {
                    label: tid.to_string(),
                    counter: false,
                    events: Vec::new(),
                });
                td.events.push((round_ns(*ts), Ek::E));
            }
            Event::Counter { pid, tid, ts, val, .. } => {
                let td = tracks.entry((*pid, *tid)).or_insert_with(|| TrackData {
                    label: tid.to_string(),
                    counter: false,
                    events: Vec::new(),
                });
                td.counter = true;
                td.events.push((round_ns(*ts), Ek::C(val.clone())));
            }
        }
    }

    // Allocate uuids: root, then procs (insertion order), then tracks (order).
    let mut uid = 0i64;
    let mut alloc = || {
        uid += 1;
        uid
    };
    let root_uuid = alloc();
    let mut proc_uuid: HashMap<i64, i64> = HashMap::new();
    for pid in procs.keys() {
        proc_uuid.insert(*pid, alloc());
    }
    let mut track_uuid: HashMap<(i64, i64), i64> = HashMap::new();
    for key in tracks.keys() {
        track_uuid.insert(*key, alloc());
    }

    // Root name = common leading dotted prefix of all group display names.
    let split_names: Vec<Vec<&str>> = procs.values().map(|(n, _)| n.split('.').collect()).collect();
    let mut common: Vec<&str> = Vec::new();
    if !split_names.is_empty() {
        let minlen = split_names.iter().map(|s| s.len()).min().unwrap();
        for i in 0..minlen {
            let c0 = split_names[0][i];
            if split_names.iter().all(|s| s[i] == c0) {
                common.push(c0);
            } else {
                break;
            }
        }
    }
    let root_name = if common.is_empty() { "trace".to_string() } else { common.join(".") };
    let strip = if common.is_empty() { String::new() } else { format!("{}.", root_name) };

    let group_name_and_rank = |name: &str, num: Option<i64>, unnumbered_idx: i64| -> (String, i64) {
        let mut short = name;
        if !strip.is_empty() && short.starts_with(strip.as_str()) {
            short = &short[strip.len()..];
        }
        let short = if short.is_empty() { name.rsplit('.').next().unwrap() } else { short };
        match num {
            Some(n) => (format!("{} {}", short, n), n),
            None => (short.to_string(), 1_000_000 + unnumbered_idx),
        }
    };

    let mut out: Vec<u8> = Vec::with_capacity(events.len() * 12 + 4096);

    // --- structure sequence: declare every track up front ---
    // Root TrackDescriptor{uuid, name, child_ordering=EXPLICIT}
    {
        let mut td = Vec::new();
        pv(&mut td, 1, root_uuid as u64);
        ps(&mut td, 2, &root_name);
        pv(&mut td, 11, ORDER_EXPLICIT);
        let mut pkt = Vec::new();
        pb(&mut pkt, 60, &td);
        pv(&mut pkt, 10, SEQ_STRUCT);
        emit(&mut out, &pkt);
    }
    let mut unnumbered = 0i64;
    for (pid, (name, num)) in &procs {
        let (gname, rank) = group_name_and_rank(name, *num, unnumbered);
        if num.is_none() {
            unnumbered += 1;
        }
        let mut td = Vec::new();
        pv(&mut td, 1, proc_uuid[pid] as u64);
        ps(&mut td, 2, &gname);
        pv(&mut td, 5, root_uuid as u64);
        pv(&mut td, 12, rank as u64);
        let mut pkt = Vec::new();
        pb(&mut pkt, 60, &td);
        pv(&mut pkt, 10, SEQ_STRUCT);
        emit(&mut out, &pkt);
    }
    for (key, tr) in &tracks {
        let mut td = Vec::new();
        pv(&mut td, 1, track_uuid[key] as u64);
        ps(&mut td, 2, &tr.label);
        pv(&mut td, 5, proc_uuid[&key.0] as u64);
        if tr.counter {
            pb(&mut td, 8, &[]); // empty CounterDescriptor marks a counter track
        }
        let mut pkt = Vec::new();
        pb(&mut pkt, 60, &td);
        pv(&mut pkt, 10, SEQ_STRUCT);
        emit(&mut out, &pkt);
    }

    // --- one data sequence per track ---
    let total_events: u64 = tracks.values().map(|tr| tr.events.len() as u64).sum();
    let progress = ProgressBar::new(total_events);
    progress.set_style(
        ProgressStyle::with_template(
            "{spinner} [{elapsed_precise}] [{bar:40}] {pos}/{len} events ({eta})",
        )
        .unwrap()
        .progress_chars("=> "),
    );
    let mut seq = 2u64;
    for (key, tr) in &tracks {
        if tr.events.is_empty() {
            continue;
        }
        let tu = track_uuid[key] as u64;
        let sid = seq;
        seq += 1;

        let mut iid_of: IndexMap<&str, u64> = IndexMap::new();
        if intern && !tr.counter {
            for (_, ek) in &tr.events {
                if let Ek::B(name) = ek {
                    if !iid_of.contains_key(name.as_str()) {
                        let i = iid_of.len() as u64 + 1;
                        iid_of.insert(name.as_str(), i);
                    }
                }
            }
            if !iid_of.is_empty() {
                let mut interned = Vec::new();
                for (n, i) in &iid_of {
                    let mut inner = Vec::new();
                    pv(&mut inner, 1, *i);
                    ps(&mut inner, 2, n);
                    pb(&mut interned, 2, &inner); // InternedData.event_names -> EventName{iid,name}
                }
                let mut pkt = Vec::new();
                pb(&mut pkt, 12, &interned);
                pv(&mut pkt, 10, sid);
                pv(&mut pkt, 13, SEQ_CLEARED);
                emit(&mut out, &pkt);
            }
        }
        let use_iid = !iid_of.is_empty();

        for (ts, ek) in &tr.events {
            let ts_u = *ts as u64;
            let mut pkt = Vec::new();
            match ek {
                Ek::B(name) => {
                    let mut te = Vec::new();
                    pv(&mut te, 9, EV_BEGIN);
                    pv(&mut te, 11, tu);
                    if use_iid {
                        pv(&mut te, 10, iid_of[name.as_str()]);
                        pv(&mut pkt, 8, ts_u);
                        pb(&mut pkt, 11, &te);
                        pv(&mut pkt, 10, sid);
                        pv(&mut pkt, 13, SEQ_NEEDS);
                    } else {
                        ps(&mut te, 23, name);
                        pv(&mut pkt, 8, ts_u);
                        pb(&mut pkt, 11, &te);
                        pv(&mut pkt, 10, sid);
                    }
                }
                Ek::E => {
                    let mut te = Vec::new();
                    pv(&mut te, 9, EV_END);
                    pv(&mut te, 11, tu);
                    pv(&mut pkt, 8, ts_u);
                    pb(&mut pkt, 11, &te);
                    pv(&mut pkt, 10, sid);
                }
                Ek::C(val) => {
                    let mut te = Vec::new();
                    pv(&mut te, 9, EV_COUNTER);
                    pv(&mut te, 11, tu);
                    match val {
                        Num::F(x) => pd(&mut te, 44, *x),
                        Num::I(n) => pv(&mut te, 30, (*n & (u64::MAX as u128)) as u64),
                    }
                    pv(&mut pkt, 8, ts_u);
                    pb(&mut pkt, 11, &te);
                    pv(&mut pkt, 10, sid);
                }
            }
            emit(&mut out, &pkt);
        }
        progress.inc(tr.events.len() as u64);
    }
    progress.finish_with_message("done");

    write_bytes(path, &out, gzip_out)
}

fn write_bytes(path: &str, data: &[u8], gzip_out: bool) -> io::Result<()> {
    let f = File::create(path)?;
    if gzip_out {
        let mut e = GzEncoder::new(BufWriter::new(f), Compression::default());
        e.write_all(data)?;
        e.finish()?;
    } else {
        let mut w = BufWriter::new(f);
        w.write_all(data)?;
        w.flush()?;
    }
    Ok(())
}

// --------------------------------------------------------------------------
// Chrome Trace-Event JSON emission (--format json). Semantically equivalent to
// the Python output; not byte-identical (key order / float formatting differ).
// --------------------------------------------------------------------------

fn json_escape(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn num_json(val: &Num) -> String {
    match val {
        Num::I(n) => format!("{}", n),
        // {:?} yields Python-repr-style floats (always with a decimal point,
        // shortest round-trip), matching json.dump's float formatting.
        Num::F(x) => format!("{:?}", x),
    }
}

fn write_json(events: &[Event], path: &str, gzip_out: bool) -> io::Result<()> {
    let mut s = String::with_capacity(events.len() * 48 + 64);
    s.push_str("{\"traceEvents\": [");
    for (i, e) in events.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        match e {
            Event::ProcName { pid, name, num } => {
                s.push_str("{\"ph\": \"M\", \"name\": \"process_name\", \"pid\": ");
                s.push_str(&pid.to_string());
                s.push_str(", \"args\": {\"name\": ");
                json_escape(name, &mut s);
                s.push_str(", \"num\": ");
                match num {
                    Some(n) => s.push_str(&n.to_string()),
                    None => s.push_str("null"),
                }
                s.push_str("}}");
            }
            Event::ThreadName { pid, tid, name } => {
                s.push_str("{\"ph\": \"M\", \"name\": \"thread_name\", \"pid\": ");
                s.push_str(&pid.to_string());
                s.push_str(", \"tid\": ");
                s.push_str(&tid.to_string());
                s.push_str(", \"args\": {\"name\": ");
                json_escape(name, &mut s);
                s.push_str("}}");
            }
            Event::Begin { pid, tid, ts, name } => {
                s.push_str("{\"ph\": \"B\", \"name\": ");
                json_escape(name, &mut s);
                s.push_str(", \"pid\": ");
                s.push_str(&pid.to_string());
                s.push_str(", \"tid\": ");
                s.push_str(&tid.to_string());
                s.push_str(", \"ts\": ");
                s.push_str(&format!("{:?}", ts));
                s.push('}');
            }
            Event::End { pid, tid, ts } => {
                s.push_str("{\"ph\": \"E\", \"pid\": ");
                s.push_str(&pid.to_string());
                s.push_str(", \"tid\": ");
                s.push_str(&tid.to_string());
                s.push_str(", \"ts\": ");
                s.push_str(&format!("{:?}", ts));
                s.push('}');
            }
            Event::Counter { pid, tid, ts, name, val } => {
                s.push_str("{\"ph\": \"C\", \"name\": ");
                json_escape(name, &mut s);
                s.push_str(", \"pid\": ");
                s.push_str(&pid.to_string());
                s.push_str(", \"tid\": ");
                s.push_str(&tid.to_string());
                s.push_str(", \"ts\": ");
                s.push_str(&format!("{:?}", ts));
                s.push_str(", \"args\": {");
                json_escape(name, &mut s);
                s.push_str(": ");
                s.push_str(&num_json(val));
                s.push_str("}}");
            }
        }
    }
    s.push_str("], \"displayTimeUnit\": \"ns\"}");
    write_bytes(path, s.as_bytes(), gzip_out)
}

// --------------------------------------------------------------------------
// --stats: per (process, thread, slice-name) total duration / count.
// --------------------------------------------------------------------------

fn print_stats(events: &[Event]) {
    let mut proc_names: HashMap<i64, String> = HashMap::new();
    let mut thread_names: HashMap<(i64, i64), String> = HashMap::new();
    for e in events {
        match e {
            Event::ProcName { pid, name, .. } => {
                proc_names.insert(*pid, name.clone());
            }
            Event::ThreadName { pid, tid, name } => {
                thread_names.insert((*pid, *tid), name.clone());
            }
            _ => {}
        }
    }

    let mut open_b: HashMap<(i64, i64), (String, f64)> = HashMap::new();
    let mut totals: IndexMap<(i64, i64, String), (f64, i64)> = IndexMap::new();
    for e in events {
        match e {
            Event::Begin { pid, tid, ts, name } => {
                open_b.insert((*pid, *tid), (name.clone(), *ts));
            }
            Event::End { pid, tid, ts } => {
                if let Some((name, ts0)) = open_b.remove(&(*pid, *tid)) {
                    let ent = totals.entry((*pid, *tid, name)).or_insert((0.0, 0));
                    ent.0 += *ts - ts0;
                    ent.1 += 1;
                }
            }
            _ => {}
        }
    }

    let mut rows: Vec<(f64, String, String, String, i64)> = totals
        .into_iter()
        .map(|((pid, tid, name), (dur, cnt))| {
            (
                dur,
                proc_names.get(&pid).cloned().unwrap_or_else(|| pid.to_string()),
                thread_names.get(&(pid, tid)).cloned().unwrap_or_else(|| tid.to_string()),
                name,
                cnt,
            )
        })
        .collect();
    rows.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.1.cmp(&a.1))
            .then_with(|| b.2.cmp(&a.2))
            .then_with(|| b.3.cmp(&a.3))
            .then_with(|| b.4.cmp(&a.4))
    });

    println!("{:>14}  {:>7}  process / thread / phase", "us total", "count");
    for (dur, proc, thread, name, cnt) in rows.iter().take(200) {
        println!("{:14.1}  {:7}  {} / {} / {}", dur, cnt, proc, thread, name);
    }
    if rows.len() > 200 {
        println!("... ({} more rows omitted)", rows.len() - 200);
    }
}

// --------------------------------------------------------------------------
// CLI
// --------------------------------------------------------------------------

struct Args {
    vcd: String,
    output: Option<String>,
    format: String,
    no_intern: bool,
    gzip: bool,
    pid_depth: usize,
    include: Option<String>,
    exclude: Option<String>,
    state_map: Vec<String>,
    derive_busy: Vec<String>,
    rename: Vec<String>,
    split_asm: bool,
    stats: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        vcd: String::new(),
        output: None,
        format: "perfetto".to_string(),
        no_intern: false,
        gzip: false,
        pid_depth: 2,
        include: None,
        exclude: None,
        state_map: Vec::new(),
        derive_busy: Vec::new(),
        rename: Vec::new(),
        split_asm: false,
        stats: false,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut got_vcd = false;
    let mut i = 0;
    while i < argv.len() {
        let arg = argv[i].clone();
        let (key, inlinev) = match arg.split_once('=') {
            Some((k, v)) if k.starts_with("--") => (k.to_string(), Some(v.to_string())),
            _ => (arg.clone(), None),
        };
        let take_val = |i: &mut usize| -> Result<String, String> {
            if let Some(v) = &inlinev {
                Ok(v.clone())
            } else {
                *i += 1;
                argv.get(*i).cloned().ok_or_else(|| format!("{}: missing value", key))
            }
        };
        match key.as_str() {
            "-o" | "--output" => a.output = Some(take_val(&mut i)?),
            "--format" => a.format = take_val(&mut i)?,
            "--no-intern" => a.no_intern = true,
            "--gzip" => a.gzip = true,
            "--pid-depth" => {
                a.pid_depth = take_val(&mut i)?.parse().map_err(|_| "--pid-depth: bad int".to_string())?
            }
            "--include" => a.include = Some(take_val(&mut i)?),
            "--exclude" => a.exclude = Some(take_val(&mut i)?),
            "--state-map" => a.state_map.push(take_val(&mut i)?),
            "--derive-busy" => a.derive_busy.push(take_val(&mut i)?),
            "--rename" => a.rename.push(take_val(&mut i)?),
            "--split-asm" => a.split_asm = true,
            "--stats" => a.stats = true,
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            _ => {
                if arg.starts_with('-') && arg != "-" {
                    return Err(format!("unknown option: {}", arg));
                }
                a.vcd = arg;
                got_vcd = true;
            }
        }
        i += 1;
    }
    if !got_vcd {
        return Err("missing input VCD (use fst2vcd for FST dumps)".to_string());
    }
    if a.format != "perfetto" && a.format != "json" {
        return Err(format!("--format must be perfetto|json (got {})", a.format));
    }
    Ok(a)
}

fn print_help() {
    eprintln!(
        "gvsoc2perfetto <vcd> [-o OUT] [--format perfetto|json] [--no-intern] [--gzip]\n\
         \x20 [--pid-depth N] [--include RE] [--exclude RE] [--state-map LEAF=V:NAME,...]\n\
         \x20 [--derive-busy LEAF_A,LEAF_B] [--rename OLD=NEW] [--split-asm] [--stats]\n\
         See scripts/gvsoc2perfetto.py for the full documentation of each flag."
    );
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    let inc = match &args.include {
        Some(s) => Some(Regex::new(s)?),
        None => None,
    };
    let exc = match &args.exclude {
        Some(s) => Some(Regex::new(s)?),
        None => None,
    };

    let (mut signals, mut changes, ps) = parse_vcd(&args.vcd, inc.as_ref(), exc.as_ref())?;

    if !args.derive_busy.is_empty() {
        let mut specs: Vec<(String, String)> = Vec::new();
        for s in &args.derive_busy {
            let parts: Vec<&str> = s.splitn(2, ',').collect();
            if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
                return Err(format!("--derive-busy expects 'LEAF_A,LEAF_B' (got {:?})", s).into());
            }
            specs.push((parts[0].to_string(), parts[1].to_string()));
        }
        let n = inject_derived_busy(&mut signals, &mut changes, &specs);
        eprintln!("--derive-busy: synthesized {} busy_derived signal(s)", n);
    }

    let state_maps: Vec<(Regex, HashMap<u128, String>)> =
        args.state_map.iter().map(|s| parse_state_map_arg(s)).collect();
    let renames: HashMap<String, String> = args
        .rename
        .iter()
        .filter_map(|s| s.split_once('=').map(|(a, b)| (a.to_string(), b.to_string())))
        .collect();

    let events = convert(&signals, &changes, ps, args.pid_depth, &state_maps, &renames, args.split_asm);

    let mut out = args.output.clone().unwrap_or_else(|| {
        if args.format == "perfetto" {
            "trace.perfetto-trace".to_string()
        } else {
            "trace.json".to_string()
        }
    });
    if args.gzip && !out.ends_with(".gz") {
        out.push_str(".gz");
    }

    if args.format == "perfetto" {
        write_perfetto(&events, &out, !args.no_intern, args.gzip)?;
    } else {
        write_json(&events, &out, args.gzip)?;
    }

    let mut track_set: HashSet<(i64, i64)> = HashSet::new();
    for e in &events {
        match e {
            Event::Begin { pid, tid, .. }
            | Event::End { pid, tid, .. }
            | Event::Counter { pid, tid, .. } => {
                track_set.insert((*pid, *tid));
            }
            _ => {}
        }
    }
    eprintln!(
        "{}: {} events on {} tracks ({} signals in VCD). Open at https://ui.perfetto.dev",
        out,
        events.len(),
        track_set.len(),
        signals.len()
    );

    if args.stats {
        print_stats(&events);
    }
    Ok(())
}

fn main() {
    std::process::exit(match run() {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("gvsoc2perfetto: error: {}", e);
            1
        }
    });
}
