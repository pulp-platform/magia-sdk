#!/usr/bin/env bash
# merge_libgcc.sh — Build a merged libgcc archive for rv32imc/ilp32 with fp16 support.
#
# Combines soft-float arithmetic from the system GCC 10 (rv32imac/ilp32)
# with fp16 conversion functions from the riscv32 GCC 15 toolchain at ~/riscv.
# The GCC 15 objects have their ELF flags patched from ilp32d to ilp32.
#
# Usage: ./scripts/merge_libgcc.sh [output_path]
#   default output: targets/magia_v2/lib/libgcc_merged.a

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

LIBGCC_OLD="/usr/lib/gcc/riscv64-unknown-elf/10.2.0/rv32imac/ilp32/libgcc.a"
LIBGCC_NEW="$HOME/riscv/lib/gcc/riscv32-unknown-elf/15.2.0/libgcc.a"
AR="$HOME/riscv/bin/riscv32-unknown-elf-ar"
OUTPUT="${1:-$REPO_DIR/targets/magia_v2/lib/libgcc_merged.a}"

for f in "$LIBGCC_OLD" "$LIBGCC_NEW" "$AR"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required file not found: $f" >&2
        exit 1
    fi
done

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

# Extract old libgcc (has soft-float arithmetic, no fp16)
mkdir "$TMPDIR/old"
(cd "$TMPDIR/old" && "$AR" x "$LIBGCC_OLD")

# Extract new libgcc (has fp16, but built for ilp32d)
mkdir "$TMPDIR/new"
(cd "$TMPDIR/new" && "$AR" x "$LIBGCC_NEW")

# Patch ELF flags on new objects: clear float-ABI bits (0x6) to get soft-float
python3 -c "
import struct, glob
for path in glob.glob('$TMPDIR/new/*.o'):
    with open(path, 'r+b') as f:
        f.seek(0x24)
        flags = struct.unpack('<I', f.read(4))[0]
        f.seek(0x24)
        f.write(struct.pack('<I', flags & ~0x6))
"

# Copy fp16 objects from new into old (overwrites any duplicates)
cp "$TMPDIR/new/extendhfsf2.o" "$TMPDIR/new/truncsfhf2.o" "$TMPDIR/old/"

# Build merged archive
mkdir -p "$(dirname "$OUTPUT")"
"$AR" rcs "$OUTPUT" "$TMPDIR/old/"*.o

echo "Created $OUTPUT"
