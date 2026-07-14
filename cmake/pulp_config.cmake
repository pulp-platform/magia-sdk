# PULP cluster configuration shared by helper modules.

# ==============================================================================
# PULP cluster settings (used by add_pulp_task)
# ==============================================================================

# Paths
set(PULP_CRT0_SRC "${CMAKE_SOURCE_DIR}/targets/${TARGET_PLATFORM}/pulp/src/pulp_crt0.S" CACHE PATH "PULP CRT0 assembly")
set(PULP_LINK_SCRIPT "${CMAKE_SOURCE_DIR}/targets/${TARGET_PLATFORM}/pulp/src/pulp_program.ld" CACHE PATH "PULP linker script")

# PULP cluster configuration
set(PULP_CORE_COUNT 8 CACHE STRING "Number of PULP cores per cluster")

# ISA/ABI — same toolchain as CV32 (GCC_PULP)
set(PULP_MARCH ${CV32_MARCH})
set(PULP_MABI  ${CV32_MABI})

# Compiler flags for PULP task binary (position-independent, bare-metal)
set(PULP_COMPILE_FLAGS
    ${PULP_MARCH}
    ${PULP_MABI}
    "-fPIC"
    "-mcmodel=medany"
    "-static"
    "-nostartfiles"
    "-nostdlib"
    "-O2"
    "-g"
    "-Wall"
    "-Wextra"
    "-Wno-unused-parameter"
    "-Wno-unused-variable"
    "-Wno-unused-function"
    "-fno-common"
    "-ffunction-sections"
    "-fdata-sections"
    "-fno-builtin"
)

set(PULP_CFLAGS_DEFINES
    "-DPULP_CORE_COUNT=${PULP_CORE_COUNT}"
    "-DPULP_TARGET"
)

set(PULP_LINK_FLAGS
    "-nostartfiles"
    "-nostdlib"
    "-Wl,-z,norelro"
    "-Wl,--allow-multiple-definition"
)

# add_pulp_task defaults
set(PULP_TASK_OUTPUT_ROOT "${CMAKE_BINARY_DIR}/bin/${TARGET_PLATFORM}")
set(PULP_TASK_BIN2HEADER_SCRIPT "${CMAKE_SOURCE_DIR}/scripts/bin2header.py")
set(PULP_TASK_EXTRACT_SYMBOLS_SCRIPT "${CMAKE_SOURCE_DIR}/scripts/extract_pulp_symbols.sh")
set(PULP_TASK_HEADER_SECTION ".pulp_binary")
set(PULP_TASK_HEADER_ADDRESS "dynamic (_pulp_binary_start)")

message(STATUS "PULP Configuration: CORES=${PULP_CORE_COUNT}, ISA=${ISA}, ABI=ilp32f")
