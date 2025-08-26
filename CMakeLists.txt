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
set(CMAKE_VERBOSE_MAKEFILE TRUE)

set(CMAKE_EXE_LINKER_FLAGS_INIT "--specs=nosys.specs")
set(CMAKE_LINK_LIBRARY_USING_WHOLE_ARCHIVE 
"-Wl,--whole-archive <LIBRARY> -Wl,--no-whole-archive"
)
set(CMAKE_LINK_LIBRARY_USING_WHOLE_ARCHIVE_SUPPORTED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

if (COMPILER STREQUAL "GCC")
    set(CMAKE_TOOLCHAIN_FILE cmake/toolchain_gcc.cmake)
elseif (COMPILER STREQUAL "LLVM")
    set(CMAKE_TOOLCHAIN_FILE cmake/toolchain_llvm.cmake)
else()
    message(FATAL_ERROR "Compiler option not supported or set. Supported options are GCC and LLVM")
endif ()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

project(magia-sdk LANGUAGES C ASM)

set(ABI ilp32)
set(ISA rv32imafc)


message(STATUS "[MAGIA-SDK] ABI                     : ${ABI}")
message(STATUS "[MAGIA-SDK] ISA                     : ${ISA}")
if (${DISASSEMBLE_LIBRARIES})
    message(STATUS "[MAGIA-SDK] DISASSEMBLE_LIBRARIES : ON")
else()
    message(STATUS "[MAGIA-SDK] DISASSEMBLE_LIBRARIES : OFF")
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

################################################################################
# Add subdirectories                                                           #
################################################################################

# Targets have to be included before the other folders to make them available
# Depending on the target, following static libraries have to added by the targets
add_subdirectory(targets)

if (COMPILER STREQUAL "LLVM")
    # VIVIANEP: Include the picolibc library
    include(${CMAKE_CURRENT_LIST_DIR}/cmake/picolibc.cmake)
endif()

# Include other subdirectories
add_subdirectory(hal)
# add_subdirectory(devices)
add_subdirectory(drivers)

################################################################################
# Testing                                                                      #
################################################################################

add_subdirectory(tests)