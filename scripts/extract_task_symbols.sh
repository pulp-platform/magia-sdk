#!/bin/bash
# Extract task symbols from ELF and append to task header
# Usage: extract_task_symbols.sh <TEST_NAME> <TASK_ELF> <TASK_HEADER> <OBJDUMP_BIN>

set -e

TEST_NAME=$1
TASK_ELF=$2
TASK_HEADER=$3
OBJDUMP=$4

GUARD_NAME=$(echo "${TEST_NAME}_TASK_BIN" | tr 'a-z' 'A-Z')

# Remove existing #endif from header
sed -i "/#endif.*__${GUARD_NAME}_H__/d" "${TASK_HEADER}"

# Add binary start address
echo "" >> "${TASK_HEADER}"
echo "/* Binary start address - defined by CV32 linker */" >> "${TASK_HEADER}"
echo "extern uint32_t _spatz_binary_start;" >> "${TASK_HEADER}"
echo "#define SPATZ_BINARY_START ((uint32_t)&_spatz_binary_start)" >> "${TASK_HEADER}"

# Add dispatcher loop address
echo "" >> "${TASK_HEADER}"
echo "/* Dispatcher loop address - for spatz_run() */" >> "${TASK_HEADER}"
DISP_ADDR=$("${OBJDUMP}" -t "${TASK_ELF}" | grep 'dispatcher_loop' | awk '{print $1}' || echo "0")
echo "#define SPATZ_DISPATCHER_LOOP (SPATZ_BINARY_START + 0x${DISP_ADDR})" >> "${TASK_HEADER}"

# Add task function entry points
echo "" >> "${TASK_HEADER}"
echo "/* Task function entry points - OFFSETS from SPATZ_BINARY_START */" >> "${TASK_HEADER}"
# Find all global functions with 'task' in their name
"${OBJDUMP}" -t "${TASK_ELF}" | grep -E "F .text.*task" | awk '{print $1, $NF}' | while read addr name; do
    # Skip __first_task (that's an internal symbol)
    if [[ "$name" != "__first_task" ]]; then
        TASK_NAME=$(echo "$name" | tr 'a-z' 'A-Z')
        echo "#define ${TASK_NAME} (SPATZ_BINARY_START + 0x${addr})" >> "${TASK_HEADER}"
    fi
done

# Close header guard
echo "" >> "${TASK_HEADER}"
echo "#endif /* __${GUARD_NAME}_H__ */" >> "${TASK_HEADER}"
