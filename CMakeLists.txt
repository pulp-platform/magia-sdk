# Copyright 2024-2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Moritz Scherer <scheremo@iis.ee.ethz.ch>
# Viviane Potocnik <vivianep@iis.ee.ethz.ch>
# Philip Wiese <wiesep@iis.ee.ethz.ch>
# Alberto Dequino <alberto.dequino@unibo.it>

cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 99)

set(CMAKE_C_COMPILER_WORKS 1)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

set(CMAKE_EXE_LINKER_FLAGS_INIT "--specs=nosys.specs")
set(CMAKE_LINK_LIBRARY_USING_WHOLE_ARCHIVE
"-Wl,--whole-archive <LIBRARY> -Wl,--no-whole-archive"
)
set(CMAKE_LINK_LIBRARY_USING_WHOLE_ARCHIVE_SUPPORTED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

if (COMPILER STREQUAL "GCC_PULP")
    set(CMAKE_TOOLCHAIN_FILE cmake/toolchain_gcc_pulp.cmake)
elseif (COMPILER STREQUAL "GCC_MULTILIB")
    set(CMAKE_TOOLCHAIN_FILE cmake/toolchain_gcc_multilib.cmake)
elseif (COMPILER STREQUAL "LLVM")
    set(CMAKE_TOOLCHAIN_FILE cmake/toolchain_llvm.cmake)
else()
    message(FATAL_ERROR "Compiler option not supported or set. Supported options are GCC and LLVM")
endif ()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

project(magia-sdk LANGUAGES C ASM)

set(EVAL 0 CACHE STRING "Enable evaluation instrumentation")
set(STALLING 0 CACHE STRING "Enable accelerator stalling")
set(FSYNC_MM 1 CACHE STRING "Use the memory-mapped fractal-sync interface")
set(IDMA_MM 1 CACHE STRING "Use the memory-mapped iDMA interface")
set(REDMULE_MM 1 CACHE STRING "Use the memory-mapped RedMulE interface")
foreach(PROFILE_OPTION PROFILE_CMP PROFILE_CMI PROFILE_CMO PROFILE_SNC)
    set(${PROFILE_OPTION} 0 CACHE STRING "Enable ${PROFILE_OPTION} profiling")
endforeach()

# Speed up rebuilds with ccache when available (opt-out via -DUSE_CCACHE=OFF).
option(USE_CCACHE "Use ccache to speed up (re)compilation if available" ON)
if (USE_CCACHE)
    find_program(CCACHE_PROGRAM ccache)
    if (CCACHE_PROGRAM)
        message(STATUS "[MAGIA-SDK] ccache                  : ${CCACHE_PROGRAM}")
        set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        set(CMAKE_ASM_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    else()
        message(STATUS "[MAGIA-SDK] ccache                  : not found (disabled)")
    endif()
endif()

set(ABI ilp32)
if (TARGET_PLATFORM STREQUAL "magia_v1")
    set(ISA rv32imac)
elseif (TARGET_PLATFORM STREQUAL "magia_v2")
    set(ISA rv32imcxgap9)
elseif (TARGET_PLATFORM STREQUAL "magia_v3")
    set(ISA rv32imcxgap9)
endif ()

if (COMPILER STREQUAL "GCC_MULTILIB")
    set(ISA rv32imc_xcvalu_xcvbi_xcvbitmanip_xcvhwlp_xcvmac_xcvmem_xcvsimd_xcvelw_zfinx_zhinxmin)
    set(CV32E40P 1)
else ()
    set(CV32E40P 0)
endif ()

message(STATUS "[MAGIA-SDK] ABI                     : ${ABI}")
message(STATUS "[MAGIA-SDK] ISA                     : ${ISA}")
if (${DISASSEMBLE_LIBRARIES})
    message(STATUS "[MAGIA-SDK] DISASSEMBLE_LIBRARIES : ON")
else()
    message(STATUS "[MAGIA-SDK] DISASSEMBLE_LIBRARIES : OFF")
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# Spatz support
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/spatz_config.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/spatz_helpers.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/bootrom_helpers.cmake)

# PULP cluster support
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/pulp_config.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/pulp_helpers.cmake)

################################################################################
# Add subdirectories                                                           #
################################################################################

# Targets have to be included before the other folders to make them available
# Depending on the target, following static libraries have to added by the targets
add_subdirectory(targets)

# Include other subdirectories
add_subdirectory(hal)
# add_subdirectory(devices)
add_subdirectory(drivers)

################################################################################
# Testing                                                                      #
################################################################################

add_subdirectory(tests)

if(DEFINED MAPS_APPLICATION_DIR)
    add_subdirectory("${MAPS_APPLICATION_DIR}" maps_application)
endif()
