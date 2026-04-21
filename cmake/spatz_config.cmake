# Spatz/CV32/Bootrom configuration shared by helper modules.

# ==============================================================================
# Spatz LLVM Toolchain [MAGIA/spatz/sw/Makefile]
# ==============================================================================
set(SPATZ_LLVM_PATH "$ENV{LLVM_INSTALL_DIR}" CACHE PATH "Path to Spatz LLVM installation")
if(NOT SPATZ_LLVM_PATH)
    set(SPATZ_LLVM_PATH "${CMAKE_SOURCE_DIR}/../llvm/install" CACHE PATH "Path to Spatz LLVM installation" FORCE)
endif()
set(SPATZ_CLANG "${SPATZ_LLVM_PATH}/bin/clang" CACHE PATH "Path to Spatz clang")
set(SPATZ_OBJCOPY "${SPATZ_LLVM_PATH}/bin/llvm-objcopy" CACHE PATH "Path to Spatz llvm-objcopy")
set(SPATZ_OBJDUMP "${SPATZ_LLVM_PATH}/bin/llvm-objdump" CACHE PATH "Path to Spatz llvm-objdump")

# ==============================================================================
# BOOTROM settings (used by add_spatz_bootrom)
# ==============================================================================
set(SPATZ_BOOTROM_SRC "${CMAKE_SOURCE_DIR}/targets/magia_v2/spatz/bootrom/spatz_init.S" CACHE PATH "Spatz bootrom source")
set(SPATZ_BOOTROM_LINK_SCRIPT "${CMAKE_SOURCE_DIR}/targets/magia_v2/spatz/bootrom/spatz_init.ld" CACHE PATH "Spatz bootrom linker script")
set(SPATZ_BOOTROM_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin/bootrom")
set(SPATZ_BOOTROM_ELF "${SPATZ_BOOTROM_OUTPUT_DIR}/spatz_init.elf")
set(SPATZ_BOOTROM_BIN "${SPATZ_BOOTROM_OUTPUT_DIR}/spatz_init.bin")
set(SPATZ_BOOTROM_DUMP "${SPATZ_BOOTROM_OUTPUT_DIR}/spatz_init.dump")

# ==============================================================================
# CV32 settings (used by add_cv32_executable_with_spatz)
# ==============================================================================

# Paths
set(CV32_CRT0_SRC "${CMAKE_SOURCE_DIR}/targets/magia_v2/src/crt0.S" CACHE PATH "CV32 CRT0 assembly")
set(MAGIA_IO_SRC "${CMAKE_SOURCE_DIR}/targets/magia_v2/src/io.c")
set(MAGIA_TARGET_INCLUDE_DIRS
    "${CMAKE_SOURCE_DIR}/targets/magia_v2/include"
    "${CMAKE_SOURCE_DIR}/targets/magia_v2/include/utils"
    "${CMAKE_SOURCE_DIR}/targets/magia_v2/include/addr_map"
    "${CMAKE_SOURCE_DIR}/targets/magia_v2/include/regs"
    "${CMAKE_SOURCE_DIR}/tests/common"
)
set(MAGIA_CV32_EXTRA_INCLUDE_DIRS
    "${CMAKE_SOURCE_DIR}/hal/include"
    "${CMAKE_SOURCE_DIR}/drivers/eventunit32/include"
)

# ISA setup [MAGIA/Makefile]
set(CV32_ARCH rv CACHE STRING "CV32 ARCH prefix")
set(CV32_XLEN 32 CACHE STRING "CV32 XLEN")
set(CV32_XTEN imcxgap9 CACHE STRING "CV32 ISA extensions")
set(CV32_ABI ilp CACHE STRING "CV32 ABI prefix")
set(CV32_XABI f CACHE STRING "CV32 ABI extension")
set(CV32_MARCH "-march=${CV32_ARCH}${CV32_XLEN}${CV32_XTEN}")
set(CV32_MABI "-mabi=${CV32_ABI}${CV32_XLEN}${CV32_XABI}")

# Compiler flags [MAGIA/Makefile: CC_OPTS]
set(CV32_COMPILE_FLAGS
    ${CV32_MARCH}
    ${CV32_MABI}
    "-D__riscv__"
    "-O2"
    "-g"
    "-Wall"
    "-Wextra"
    "-Wno-unused-parameter"
    "-Wno-unused-variable"
    "-Wno-unused-function"
    "-Wundef"
    "-fdata-sections"
    "-ffunction-sections"
    "-MMD"
    "-MP"
)

# Linker flags [MAGIA/Makefile: LD_OPTS]
set(CV32_LINK_FLAGS
    ${CV32_MARCH}
    ${CV32_MABI}
    "-D__riscv__"
    "-MMD"
    "-MP"
    "-nostartfiles"
    "-nostdlib"
    "-Wl,--gc-sections"
)

# ==============================================================================
# SPATZ settings (used by add_spatz_task)
# ==============================================================================

# Paths
set(SPATZ_CRT0_SRC "${CMAKE_SOURCE_DIR}/targets/magia_v2/spatz/src/spatz_crt0.S" CACHE PATH "Spatz CRT0 assembly")
set(SPATZ_LINK_SCRIPT "${CMAKE_SOURCE_DIR}/targets/magia_v2/spatz/src/spatz_program.ld" CACHE PATH "Spatz linker script")

# Spatz arch configuration options [MAGIA/Makefile]
set(SPATZ_RVD 0 CACHE INT "0: 32-bit TCDM w/ ELEN=32, 1: 64-bit TCDM w/ ELEN=64")
set(SPATZ_VLEN 256 CACHE INT "Vector length in bits (128, 256, 512, ...)")
set(SPATZ_NRVREG 32 CACHE INT "Number of vector registers - RISC-V standar=32")
set(SPATZ_NR_VRF_BANKS 4 CACHE INT "Number of VRF banks (banking parallelism: 2, 4, 8)")
set(SPATZ_N_IPU 1 CACHE INT "Number of Integer Processing Units (1-8)")
set(SPATZ_N_FPU 4 CACHE INT "Number of Floating Point Units (1-8)")
set(SPATZ_NR_PARALLEL_INSTR 4 CACHE INT "Number of parallel vector instructions (scoreboard depth)")
set(SPATZ_XDIVSQRT 0 CACHE INT "0: FP div/sqrt disabled, 1: enabled")
set(SPATZ_XDMA 0 CACHE INT "0: DMA disabled, 1: enabled")
set(SPATZ_RVF 1 CACHE INT "0: single-precision FP disabled, 1: enabled")
set(SPATZ_RVV 1 CACHE INT "0: vector extension disabled, 1: enabled")

# ISA setup [MAGIA/spatz/sw/Makefile]
set(SPATZ_ARCH rv CACHE STRING "Spatz ARCH prefix")
set(SPATZ_XLEN 32 CACHE STRING "Spatz XLEN")
set(SPATZ_ABI ilp CACHE STRING "Spatz ABI prefix")

# Build ISA string dynamically [MAGIA/spatz/sw/Makefile]
set(SPATZ_XTEN "ima")
if(SPATZ_RVD)
    set(SPATZ_XTEN "${SPATZ_XTEN}fd")
    set(SPATZ_XABI "32d")
elseif(SPATZ_RVF)
    set(SPATZ_XTEN "${SPATZ_XTEN}f")
    set(SPATZ_XABI "32f")
else()
    set(SPATZ_XABI "32")
endif()
if(SPATZ_RVV)
    set(SPATZ_XTEN "${SPATZ_XTEN}v")
endif()
set(SPATZ_XTEN "${SPATZ_XTEN}_zfh")
set(SPATZ_MARCH "-march=${SPATZ_ARCH}${SPATZ_XLEN}${SPATZ_XTEN}")
set(SPATZ_MABI "-mabi=${SPATZ_ABI}${SPATZ_XABI}")

# Compiler and linker flags [MAGIA/spatz/sw/Makefile]
set(SPATZ_ARCH_FLAGS
    "--target=riscv32"
    ${SPATZ_MARCH}
    ${SPATZ_MABI}
    "-menable-experimental-extensions"
)
set(SPATZ_COMPILE_FLAGS
    ${SPATZ_ARCH_FLAGS}
    "-static"
    "-nostartfiles"
    "-nostdlib"
    "-O3"
    "-g"
    "-Wall"
    "-Wextra"
    "-Wno-incompatible-function-pointer-types"
    "-fno-common"
    "-ffunction-sections"
    "-fdata-sections"
    "-fPIC"
    "-fno-builtin"
    "-mcmodel=medany"
)
set(SPATZ_CFLAGS_DEFINES
    "-DSPATZ_RVD=${SPATZ_RVD}"
    "-DSPATZ_VLEN=${SPATZ_VLEN}"
    "-DSPATZ_N_IPU=${SPATZ_N_IPU}"
    "-DSPATZ_N_FPU=${SPATZ_N_FPU}"
    "-DSPATZ_XDIVSQRT=${SPATZ_XDIVSQRT}"
    "-DSPATZ_XDMA=${SPATZ_XDMA}"
    "-DSPATZ_RVF=${SPATZ_RVF}"
    "-DSPATZ_RVV=${SPATZ_RVV}"
)
set(SPATZ_LINK_FLAGS
    "-fuse-ld=${SPATZ_LLVM_PATH}/bin/ld.lld"
    "-Wl,-z,norelro"
    "-Wl,--allow-multiple-definition"
)

# add_spatz_task defaults
set(SPATZ_TASK_DEFINE "-DSPATZ_TARGET")
set(SPATZ_TASK_OUTPUT_ROOT "${CMAKE_BINARY_DIR}/bin/spatz_on_magia")
set(SPATZ_TASK_BIN2HEADER_SCRIPT "${CMAKE_SOURCE_DIR}/scripts/bin2header.py")
set(SPATZ_TASK_EXTRACT_SYMBOLS_SCRIPT "${CMAKE_SOURCE_DIR}/scripts/extract_task_symbols.sh")
set(SPATZ_TASK_HEADER_SECTION ".spatz_binary")
set(SPATZ_TASK_HEADER_ADDRESS "dynamic (_spatz_binary_start)")

message(STATUS "Spatz Configuration: LLVM=${SPATZ_LLVM_PATH}, ISA=rv32${SPATZ_XTEN}, VLEN=${SPATZ_VLEN}")
