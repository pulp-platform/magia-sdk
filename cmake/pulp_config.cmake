# PULP cluster configuration shared by helper modules.

# ==============================================================================
# PULP cluster settings (used by add_pulp_task)
# ==============================================================================

# Paths
set(PULP_CRT0_SRC "${CMAKE_SOURCE_DIR}/targets/${TARGET_PLATFORM}/pulp/src/pulp_crt0.S" CACHE PATH "PULP CRT0 assembly")
set(PULP_LINK_SCRIPT "${CMAKE_SOURCE_DIR}/targets/${TARGET_PLATFORM}/pulp/src/pulp_program.ld" CACHE PATH "PULP linker script")

# PULP cluster configuration
set(PULP_CORE_COUNT 8 CACHE STRING "Number of PULP cores per cluster")

# ISA/ABI — same toolchain as CV32 Controller
set(PULP_ARCH rv CACHE STRING "PULP32 ARCH prefix")
set(PULP_XLEN 32 CACHE STRING "PULP32 XLEN")
set(PULP_XTEN imc_xcvalu_xcvbi_xcvbitmanip_xcvhwlp_xcvmac_xcvmem_xcvsimd_xcvelw_zfinx_zhinxmin CACHE STRING "PULP32 ISA extensions")
set(PULP_ABI ilp CACHE STRING "PULP32 ABI prefix")
set(PULP_MARCH "-march=${PULP_ARCH}${PULP_XLEN}${PULP_XTEN}")
set(PULP_MABI "-mabi=${PULP_ABI}${PULP_XLEN}")

# Compiler and linker flags [MAGIA/sw/kernel_pulp/Makefile]
set(PULP_ARCH_FLAGS
    ${PULP_MARCH}
    ${PULP_MABI}
)
set(PULP_COMPILE_FLAGS
    ${PULP_ARCH_FLAGS}
    "-O2"
    "-g"
    "-Wall"
    "-Wextra"
    "-Wno-unused-parameter"
    "-Wno-unused-variable"
    "-Wno-unused-function"
    "-Wundef"
    "-ffunction-sections"
    "-fdata-sections"
    "-fPIC"
    "-mcmodel=medany"
    "-fno-builtin"
    "-fno-jump-tables"
    "-fno-common"
    "-msmall-data-limit=0"
    "-DPULP_CORE_COUNT=${PULP_CORE_COUNT}"
    "-DCV32E40P"
    "-nostartfiles"
    "-nostdlib"
    "-U__riscv__"
    "-static"
)
set(PULP_CFLAGS_DEFINES
    "-DPULP_CORE_COUNT=${PULP_CORE_COUNT}"
    "-DPULP_TARGET"
    "-DCV32E40P"
    # "-save-temps"
)
 set(PULP_LINK_FLAGS
    ${PULP_ARCH_FLAGS}
    "-nostartfiles"
    "-nostdlib"
    "-Wl,--gc-sections"
    "-Wl,--allow-multiple-definition"
    "-T${PULP_LINK_SCRIPT}"
    "-flto"
    "-Wl,-z,norelro"
)
# # Compiler flags for PULP task binary (position-independent, bare-metal)
# set(PULP_COMPILE_FLAGS
#     ${PULP_MARCH}
#     ${PULP_MABI}
#     "-fPIC"
#     "-mcmodel=medany"
#     "-static"
#     "-nostartfiles"
#     "-nostdlib"
#     "-O2"
#     "-g"
#     "-Wall"
#     "-Wextra"
#     "-Wno-unused-parameter"
#     "-Wno-unused-variable"
#     "-Wno-unused-function"
#     "-Wundef"
#     "-fno-common"
#     "-ffunction-sections"
#     "-fdata-sections"
#     "-fno-builtin"
#     "-fno-jump-tables"
#     "-msmall-data-limit=0"
# )

# set(PULP_CFLAGS_DEFINES
#     "-DPULP_CORE_COUNT=${PULP_CORE_COUNT}"
#     "-DPULP_TARGET"
#     "-DCV32E40P=${CV32E40P}"
#     "-save-temps"
# )

# set(PULP_LINK_FLAGS
#     "-nostartfiles"
#     "-nostdlib"
#     "-Wl,-z,norelro"
#     "-Wl,--allow-multiple-definition"
#     "-flto"
#     "-Wl,--gc-sections"
#     "-Wl,--undefined=hello_task"
# )

# add_pulp_task defaults
set(PULP_TASK_OUTPUT_ROOT "${CMAKE_BINARY_DIR}/bin/${TARGET_PLATFORM}")
set(PULP_TASK_BIN2HEADER_SCRIPT "${CMAKE_SOURCE_DIR}/scripts/bin2header.py")
set(PULP_TASK_EXTRACT_SYMBOLS_SCRIPT "${CMAKE_SOURCE_DIR}/scripts/extract_pulp_symbols.sh")
set(PULP_TASK_HEADER_SECTION ".pulp_binary")
set(PULP_TASK_HEADER_ADDRESS "dynamic (_pulp_binary_start)")

message(STATUS "PULP Configuration: CORES=${PULP_CORE_COUNT}, ISA=${ISA}, ABI=ilp32f")
