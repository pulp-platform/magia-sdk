#!/bin/bash
# Extract PULP binary symbols and append to task header.
# Usage: extract_pulp_symbols.sh <TEST_NAME> <TASK_ELF> <TASK_HEADER> <OBJDUMP_BIN>
#
# The PULP header is named <TEST_NAME>_task_bin.h with guard __<TEST_NAME>_TASK_BIN_H__,
# matching the --name <TEST_NAME>_task_bin argument passed to bin2header.py.

set -e

TEST_NAME=$1
TASK_ELF=$2
TASK_HEADER=$3
OBJDUMP=$4

GUARD_NAME=$(echo "${TEST_NAME}_TASK_BIN" | tr 'a-z' 'A-Z')

# Remove existing #endif from header
sed -i "/#endif.*__${GUARD_NAME}_H__/d" "${TASK_HEADER}"

# Add _pulp_binary_start — defined by the CV32 linker (link.ld)
echo "" >> "${TASK_HEADER}"
echo "/* Binary start address - defined by CV32 linker */" >> "${TASK_HEADER}"
echo "extern uint32_t _pulp_binary_start;" >> "${TASK_HEADER}"
echo "#define PULP_BINARY_START ((uint32_t)&_pulp_binary_start)" >> "${TASK_HEADER}"

# Add task function entry points (all global functions with 'task' in their name)
echo "" >> "${TASK_HEADER}"
echo "/* Task function entry points - OFFSETS from PULP_BINARY_START */" >> "${TASK_HEADER}"
"${OBJDUMP}" -t "${TASK_ELF}" | awk '$2 == "g" && $NF ~ /task/ {print $1, $NF}' | while read addr name; do
    if [[ "$name" != "__first_task" ]]; then
    TASK_NAME=$(echo "$name" | tr 'a-z' 'A-Z')
    echo "#define ${TASK_NAME} (PULP_BINARY_START + 0x${addr})" >> "${TASK_HEADER}"
    fi
done

# Close header guard
echo "" >> "${TASK_HEADER}"
echo "#endif /* __${GUARD_NAME}_H__ */" >> "${TASK_HEADER}"
