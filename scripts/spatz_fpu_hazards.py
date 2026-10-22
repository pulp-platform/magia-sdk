"""Find FP instructions of Spatz tasks that need SPATZ_FPU_FENCE.

usage: spatz_fpu_hazards.py <llvm-objdump> <task.elf>...

FPnew completes out of order across op groups of different latency (MAGIA: ADDMUL 3, CONV 2,
NONCOMP 1, DOTP 5). An FP instruction issued to the VFU with a lower latency than the FP
instruction before it can overtake its last result and deadlock the chaining. Dataflow over each
function's CFG of the latency L of the FP instructions possibly in flight:
  vector FP op of latency l:  hazard if l < L (the fence drains first); then L = l
  FP reduction:               no hazard (the VFU waits for the FPU to drain before a reduction,
                              Reduction_Wait); then L = its latency
  scalar FP op of latency l:  L = max(L, l)  (no VRF dependency of its own, but it sits in the
                              same pipeline: a later vector op still has to wait for the older one)
  VFU integer op:             L = 0  (the VFU drains the FPU before switching to the IPU)
  task entry (_start, *_task): L = 0 (the previous task's last stores waited for its results)
  other function entry:       L = max of L at its call sites; after a call, L = callee's exit L
Prints `file:line  addr  insn  function` per hazard, de-duplicated by file:line + mnemonic.

Also reports a VFU instruction with a scalar result (vmv.x.s, vfmv.f.s, e.g.
the fence itself) whose next VFU instruction is a reduction: the VFU accepts the reduction while
the scalar result is still in the IPU result register, and the result is dropped without a
response. Paths on which an instruction reads or writes the result register in between are safe:
it stalls until the response. Printed with `prev_lat=rsp` and the address of the reduction.
"""
import re
import subprocess
import sys

MAXLAT = 5
ADDMUL = re.compile(r'^v?fw?(add|sub|rsub|mul|n?macc|n?msac|n?madd|n?msub)\b|^vf(w)?red(u|o)sum\b|'
                    r'^f(add|sub|mul|n?madd|n?msub)\.')
DOTP = re.compile(r'^vfw?dotp')
CONV = re.compile(r'^v?f[wn]?cvt\b|^vf[wn]?cvt\.|^fcvt\.')
NONCOMP = re.compile(r'^vf(min|max|sgnj|sgnjn|sgnjx|class|redmin|redmax)\b|^vmf(eq|ne|lt|le|gt|ge)\b|'
                     r'^f(min|max|sgnj|sgnjn|sgnjx|eq|lt|le|class)\.|^f(mv|neg|abs)\.[hs]$')
# VFU integer ops (vmv.x.s and vfmv.f.s are VADD on the VFU, see spatz_decoder.sv)
IPU = re.compile(r'^v(add|sub|rsub|and|or|xor|sll|srl|sra|n?srl|mul|mulh|mulhu|madd|macc|nmsac|nmsub|'
                 r'min|minu|max|maxu|ms(eq|ne|lt|ltu|le|leu|gt|gtu)|redsum|redmax|redmin|merge|mv\.x\.s)\b|'
                 r'^vfmv\.f\.s$|^vmv\.x\.s$')
# vector reductions (spatz_decoder.sv: is_reduction)
REDUCTION = re.compile(r'^vf?w?red')
# VFU instructions with a scalar result sent back to the Snitch
SCALAR_RSP = re.compile(r'^(vmv\.x\.s|vfmv\.f\.s)$')
# vector instructions that do not go to the VFU (VLSU, slide unit, controller)
NON_VFU = re.compile(r'^(vl[0-9a-z]*e|vs[0-9a-z]*e|vset|vf?slide|vfmv\.v\.f$|vfmv\.s\.f$)')
BRANCH = re.compile(r'^(beq|bne|blt|bge|bltu|bgeu|beqz|bnez|blez|bgez|bltz|bgtz|bgt|ble|bgtu|bleu)$')


def latency(mn):
    """(vector?, latency) of an FP op, None otherwise."""
    if DOTP.match(mn):
        return (True, 5)
    for rx, lat in ((ADDMUL, 3), (CONV, 2), (NONCOMP, 1)):
        if rx.match(mn):
            return (mn.startswith('v'), lat)
    return None


def parse(objdump, elf):
    out = subprocess.run([objdump, '-d', '-l', '--no-show-raw-insn', '--mattr=+m,+a,+f,+v,+zfh', elf],
                         capture_output=True, text=True).stdout
    funcs, cur, loc, func = {}, None, '?', None
    for line in out.splitlines():
        m = re.match(r'^([0-9a-f]+) <([^>]+)>:', line)
        if m:
            if not m.group(2).startswith('.L'):
                func = m.group(2)
                funcs[func] = []
            continue
        m = re.match(r'^; (/\S+:\d+)', line)
        if m:
            loc = m.group(1)
            continue
        m = re.match(r'^\s+([0-9a-f]+):\s+(\S+)\s*(.*)$', line)
        if m and func:
            a, mn, ops = int(m.group(1), 16), m.group(2), m.group(3).strip()
            # auipc rX, hi; jalr [rd,] lo(rX): PC-relative call, resolve it to `jal 0x<target>`
            j = re.match(r'^(?:(\w+),\s*)?(-?\d+)\((\w+)\)$', ops) if mn == 'jalr' else None
            prev = funcs[func][-1] if funcs[func] else None
            if j and prev and prev[1] == 'auipc' and prev[2].startswith(j.group(3) + ','):
                hi = int(prev[2].split(',')[1], 0)
                tgt = (prev[0] + (hi << 12) + int(j.group(2))) & 0xffffffff
                mn, ops = ('jal' if (j.group(1) or 'ra') != 'zero' else 'j'), '0x%x' % tgt
            funcs[func].append((a, mn, ops, loc))
    return funcs


def call_target(mn, ops, addr2func):
    if mn not in ('jal', 'call', 'j', 'tail'):
        return None
    m = re.search(r'<([^>+]+)>', ops)
    if m and not m.group(1).startswith('.L'):
        return m.group(1)
    m = re.search(r'0x([0-9a-f]+)', ops)
    return addr2func.get(int(m.group(1), 16)) if m else None


def analyze(func, insns, entry, exits, addr2func, calls):
    idx = {a: i for i, (a, _, _, _) in enumerate(insns)}
    succ = []
    for i, (a, mn, ops, _) in enumerate(insns):
        tgt = re.search(r'0x([0-9a-f]+)', ops)
        t = idx.get(int(tgt.group(1), 16)) if tgt else None
        if BRANCH.match(mn):
            succ.append([s for s in (i + 1, t) if s is not None and s < len(insns)])
        elif mn == 'j':
            succ.append([t] if t is not None else [])
        elif mn in ('ret', 'jr', 'mret') or (mn == 'jalr' and ops.startswith('zero')):
            succ.append([])
        else:
            succ.append([i + 1] if i + 1 < len(insns) else [])
    state = [-1] * len(insns)
    state[0] = entry
    exit_l = 0
    work = [0]
    hazards = {}
    while work:
        i = work.pop()
        L = state[i]
        a, mn, ops, loc = insns[i]
        lat = latency(mn)
        if lat and lat[0]:
            if lat[1] < L and not REDUCTION.match(mn):
                hazards[i] = L
            out = lat[1]
        elif lat:
            out = max(L, lat[1])
        elif IPU.match(mn):
            out = 0
        elif mn in ('jal', 'call') or (mn == 'jalr' and not ops.startswith('zero')):
            callee = call_target(mn, ops, addr2func)
            if callee is not None:
                calls[callee] = max(calls.get(callee, 0), L)
            out = exits.get(callee, MAXLAT) if callee is not None else MAXLAT
        elif mn in ('j', 'tail') and not succ[i]:
            callee = call_target(mn, ops, addr2func)  # tail call
            if callee is not None:
                calls[callee] = max(calls.get(callee, 0), L)
            out = exits.get(callee, MAXLAT) if callee is not None else MAXLAT
        else:
            out = L
        if not succ[i]:
            exit_l = max(exit_l, out)
        for s in succ[i]:
            if out > state[s]:
                state[s] = out
                work.append(s)
    # a scalar-result VFU instruction followed by a reduction as the next VFU instruction
    for i, (a, mn, ops, loc) in enumerate(insns):
        if not SCALAR_RSP.match(mn):
            continue
        rd = ops.split(',')[0].strip()
        seen, todo = set(), list(succ[i])
        while todo:
            j = todo.pop()
            if j in seen:
                continue
            seen.add(j)
            mj = insns[j][1]
            # an instruction that reads or writes rd waits for the response
            if rd in re.findall(r'[\w.]+', insns[j][2]):
                continue
            if mj.startswith('v') and not NON_VFU.match(mj):
                if REDUCTION.match(mj):
                    hazards[j] = 'rsp'
                continue
            todo.extend(succ[j])
    return [(insns[i], L) for i, L in sorted(hazards.items())], exit_l


def main():
    objdump, elfs = sys.argv[1], sys.argv[2:]
    seen = {}
    for elf in elfs:
        funcs = {f: i for f, i in parse(objdump, elf).items() if i}
        addr2func = {i[0][0]: f for f, i in funcs.items()}
        is_entry = {f: f == '_start' or f.endswith('_task') for f in funcs}
        entries = {f: 0 for f in funcs}
        exits = {f: 0 for f in funcs}
        for _ in range(20):
            calls, new_exits, results = {}, {}, {}
            for f, insns in funcs.items():
                results[f], new_exits[f] = analyze(f, insns, entries[f], exits, addr2func, calls)
            new_entries = {f: 0 if is_entry[f] else calls.get(f, MAXLAT) for f in funcs}
            if new_entries == entries and new_exits == exits:
                break
            entries, exits = new_entries, new_exits
        for func in funcs:
            for (a, mn, ops, loc), L in results[func]:
                key = (loc, mn)
                if key not in seen:
                    seen[key] = (a, mn, ops, func, L, elf)
    for (loc, mn), (a, _, ops, func, L, elf) in sorted(seen.items()):
        print('%s\t%x\t%s %s\t%s\tprev_lat=%s' % (loc, a, mn, ops, func, L))


main()
