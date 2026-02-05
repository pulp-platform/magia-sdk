# Spatz task global config

set(SPATZ_LLVM_PATH "/opt/riscv/spatz-14-llvm" CACHE PATH "Path to Spatz LLVM installation")

# Toolchain paths [MAGIA/spatz/sw/Makefile]
find_program(SPATZ_CLANG clang PATHS "${SPATZ_LLVM_PATH}/bin" REQUIRED)
find_program(SPATZ_OBJCOPY llvm-objcopy PATHS "${SPATZ_LLVM_PATH}/bin" REQUIRED)
find_program(SPATZ_OBJDUMP llvm-objdump PATHS "${SPATZ_LLVM_PATH}/bin" REQUIRED)

# Spatz configuration options [MAGIA/Makefile]
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

# Spatz ISA configuration [MAGIA/spatz/sw/Makefile]
set(SPATZ_ARCH rv CACHE STRING "Spatz ARCH prefix (e.g. rv)")
set(SPATZ_XLEN 32 CACHE STRING "Spatz XLEN (32 or 64)")
set(SPATZ_ABI ilp CACHE STRING "Spatz ABI prefix (e.g. ilp)")

# Build isa string dynamically [MAGIA/spatz/sw/Makefile]
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

# CV32 configuration options [MAGIA/Makefile]
set(CV32_ARCH rv CACHE STRING "CV32 ARCH prefix (e.g. rv)")
set(CV32_XLEN 32 CACHE STRING "CV32 XLEN (32 or 64)")
set(CV32_XTEN imfcxpulpv2 CACHE STRING "CV32 XTEN (ISA extensions)")
set(CV32_ABI ilp CACHE STRING "CV32 ABI prefix (e.g. ilp)")
set(CV32_XABI f CACHE STRING "CV32 XABI (32, 32f, 32d)")
set(CV32_MARCH "-march=${CV32_ARCH}${CV32_XLEN}${CV32_XTEN}")
set(CV32_MABI "-mabi=${CV32_ABI}${CV32_XLEN}${CV32_XABI}")

# Architecture and Compiler flags [MAGIA/spatz/sw/Makefile]
set(SPATZ_COMPILE_FLAGS
    "--target=riscv32"
    ${SPATZ_MARCH}
    ${SPATZ_MABI}
    "-menable-experimental-extensions"
    "-static"
    "-nostartfiles"
    "-nostdlib"
    "-O2"
    "-g"
    "-Wall"
    "-Wextra"
    "-Wno-incompatible-function-pointer-types"
    "-fno-common"
    "-ffunction-sections"
    "-fdata-sections"
    "-fPIC"
    "-fno-builtin-memset"
    "-fno-builtin-memcpy"
    "-mcmodel=medany"
)

# Spatz defines [MAGIA/spatz/sw/Makefile]
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

# Linker flags [MAGIA/spatz/sw/Makefile]
set(SPATZ_LINK_FLAGS
    "-fuse-ld=${SPATZ_LLVM_PATH}/bin/ld.lld"
    "-Wl,-z,norelro"
    "-Wl,--allow-multiple-definition"
)

set(SPATZ_CRT0_SRC "${CMAKE_SOURCE_DIR}/targets/magia_v2/spatz/src/spatz_crt0.S" CACHE PATH "Spatz CRT0 assembly")
set(SPATZ_LINK_SCRIPT "${CMAKE_SOURCE_DIR}/targets/magia_v2/spatz/src/spatz_program.ld" CACHE PATH "Spatz linker script")
set(CV32_CRT0_SRC "${CMAKE_SOURCE_DIR}/targets/magia_v2/src/crt0.S" CACHE PATH "CV32 CRT0 assembly")
set(CV32_LINK_SCRIPT "${CMAKE_SOURCE_DIR}/targets/magia_v2/link.ld" CACHE PATH "CV32 linker script")

# TODO: check if needed
# Mesh tile config from root Makefile
set(TILES 2 CACHE INT "Mesh dimension (e.g. 2 implies 2x2 grid = 4 tiles)")


# Log info
message(STATUS "Spatz Configuration:")
message(STATUS "  LLVM Path: ${SPATZ_LLVM_PATH}")
message(STATUS "  ISA: rv32${SPATZ_XTEN}")
message(STATUS "  VLEN: ${SPATZ_VLEN} bits")
message(STATUS "  Mesh: ${TILES}x${TILES} (${TILES_2} tiles)")
