#!/bin/bash
# Extract PULP binary symbols and append to task header.
# Usage: extract_pulp_symbols.sh <TEST_NAME> <TASK_ELF> <TASK_HEADER> <OBJDUMP_BIN>
#
# The PULP header is named <TEST_NAME>_bin.h with guard __<TEST_NAME>_BIN_H__,
# matching the --name <TEST_NAME>_bin argument passed to bin2header.py.

set -e

TEST_NAME=$1
TASK_ELF=$2
TASK_HEADER=$3
OBJDUMP=$4

GUARD_NAME=$(echo "${TEST_NAME}_BIN" | tr 'a-z' 'A-Z')

# Remove existing #endif from header
sed -i "/#endif.*__${GUARD_NAME}_H__/d" "${TASK_HEADER}"

# Add _pulp_binary_start — defined by the CV32 linker (link.ld)
echo "" >> "${TASK_HEADER}"
echo "/* Binary start address - defined by CV32 linker */" >> "${TASK_HEADER}"
echo "extern uint32_t _pulp_binary_start;" >> "${TASK_HEADER}"
echo "#define PULP_BINARY_START ((uint32_t)&_pulp_binary_start)" >> "${TASK_HEADER}"

# Close header guard
echo "" >> "${TASK_HEADER}"
echo "#endif /* __${GUARD_NAME}_H__ */" >> "${TASK_HEADER}"
