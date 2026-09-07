// Copyright (C) 2018-2026 ETH Zurich and University of Bologna
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// SPDX-License-Identifier: Apache-2.0
//
// Replaces parse_s19.pl + s19tomem.py: parses an SREC (S19) object dump
// directly and splits it into instruction/data $readmemh stimuli, with no
// intermediate text format.
//
// Usage: s19tomem <input.s19> [instr_out.txt] [data_out.txt]
//
// Memory layout (must match sw/kernel/link.ld):
//   Instructions start at 0xcc00_0000
//   Data         starts at 0xcc01_0000
//   Stack        starts at 0x0001_0000 (reserved, never populated from S19)

use std::env;
use std::fs;
use std::io::{self, Write, BufWriter};
use std::process;

const MEM_START: u32 = 0xcc00_0000;
const INSTR_SIZE: u32 = 0x8000;
const INSTR_END: u32 = MEM_START + INSTR_SIZE;
const DATA_BASE: u32 = MEM_START + 0x10000;
const DATA_SIZE: u32 = 0x60000;
const DATA_END: u32 = DATA_BASE + DATA_SIZE;
const STACK_SIZE: u32 = 0x10000;

const INSTR_MEM_SIZE: usize = INSTR_SIZE as usize;
const DATA_MEM_SIZE: usize = (DATA_SIZE + STACK_SIZE) as usize;

// Value of one ASCII hex digit, or None if `b` isn't 0-9/A-F/a-f.
fn hex_nibble(b: u8) -> Option<u32> {
    match b {
        b'0'..=b'9' => Some((b - b'0') as u32),
        b'A'..=b'F' => Some((b - b'A' + 10) as u32),
        b'a'..=b'f' => Some((b - b'a' + 10) as u32),
        _ => None,
    }
}

// Parses a 2-hex-char string into the byte it represents (e.g. "3F" -> 0x3F).
fn hex_byte(s: &[u8]) -> Option<u8> {
    Some(((hex_nibble(s[0])? << 4) | hex_nibble(s[1])?) as u8)
}

// Parses a hex string of any length into a u32, most-significant digit first.
fn hex_u32(s: &[u8]) -> Option<u32> {
    let mut v = 0u32;
    for chunk in s {
        v = (v << 4) | hex_nibble(*chunk)?;
    }
    Some(v)
}

// Parses one S3 SREC line:
//
//   S3 CC AAAAAAAA DD DD ... KK
//      |  |        |       checksum (1 byte, ignored, same as before)
//      |  |        payload data bytes (0 or more)
//      |  32-bit address, big-endian, 8 hex chars
//      byte count of everything after this field (4 addr + N data + 1 checksum)
//
// Calls store(address, byte) for every data byte on the line. Anything that
// isn't a well-formed S3 record (other record types S0/S1/S2/S5/S7/S8/S9,
// or a malformed line) is silently skipped.
fn parse_s3_line(line: &[u8], mut store: impl FnMut(u32, u8)) {
    if line.len() < 4 || &line[0..2] != b"S3" {
        return;
    }
    let count = match hex_byte(&line[2..4]) {
        Some(c) => c as usize,
        None => return,
    };
    if count < 5 || line.len() < 4 + count * 2 {
        return;
    }
    let data_len = count - 5;
    let addr = match hex_u32(&line[4..12]) {
        Some(a) => a,
        None => return,
    };
    for i in 0..data_len {
        let off = 12 + i * 2;
        let byte = match hex_byte(&line[off..off + 2]) {
            Some(b) => b,
            None => continue,
        };
        store(addr.wrapping_add(i as u32), byte);
    }
}

// Writes one $readmemh stimulus file: a "@<hex address>" header line
// followed by one two-hex-digit byte per line. 
// `trailing_newline` selects whether there is a trailing newline
// after the last byte.
fn write_mem_file(path: &str, base: u32, mem: &[u8], trailing_newline: bool) -> io::Result<()> {
    let f = fs::File::create(path)?;
    let mut w = BufWriter::new(f);
    writeln!(w, "@{:08x}", base)?;
    for (i, b) in mem.iter().enumerate() {
        if !trailing_newline && i == mem.len() - 1 {
            write!(w, "{:02x}", b)?;
        } else {
            writeln!(w, "{:02x}", b)?;
        }
    }
    w.flush()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} <input.s19> [instr_out.txt] [data_out.txt]", args[0]);
        process::exit(1);
    }
    let in_path = &args[1];
    let instr_path = args.get(2).map(String::as_str).unwrap_or("stim_instr.txt");
    let data_path = args.get(3).map(String::as_str).unwrap_or("stim_data.txt");

    let bytes = match fs::read(in_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: could not read '{}': {}", in_path, e);
            process::exit(1);
        }
    };

    // Simulated instruction/data memories, zero-filled, to be dumped below.
    let mut instr_mem = vec![0u8; INSTR_MEM_SIZE];
    let mut data_mem = vec![0u8; DATA_MEM_SIZE];

    for line in bytes.split(|&b| b == b'\n') {
        let line = trim(line);
        if line.is_empty() {
            continue;
        }
        // Route each decoded byte to whichever memory its address falls
        // into (or drop it if it's outside both ranges).
        parse_s3_line(line, |addr, byte| {
            if addr >= DATA_BASE && addr < DATA_END {
                data_mem[(addr - DATA_BASE) as usize] = byte;
            } else if addr >= MEM_START && addr < INSTR_END {
                instr_mem[(addr - MEM_START) as usize] = byte;
            }
        });
    }

    if let Err(e) = write_mem_file(instr_path, MEM_START, &instr_mem, true) {
        eprintln!("error: could not write '{}': {}", instr_path, e);
        process::exit(1);
    }
    if let Err(e) = write_mem_file(data_path, DATA_BASE, &data_mem, false) {
        eprintln!("error: could not write '{}': {}", data_path, e);
        process::exit(1);
    }
}

// Trims leading/trailing whitespace (spaces, '\r' from CRLF endings, etc.)
// from a line without copying it.
fn trim(line: &[u8]) -> &[u8] {
    let mut start = 0;
    let mut end = line.len();
    while start < end && (line[start] as char).is_whitespace() {
        start += 1;
    }
    while end > start && (line[end - 1] as char).is_whitespace() {
        end -= 1;
    }
    &line[start..end]
}
