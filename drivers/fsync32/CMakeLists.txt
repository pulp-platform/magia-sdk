# Copyright 2024-2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Viviane Potocnik <vivianep@iis.ee.ethz.ch>
# Alberto Dequino <alberto.dequino@unibo.it>

# 32-bit FractalSync Driver CMake Configuration

add_library(fsync_lib STATIC)

file(GLOB_RECURSE SOURCES
  "src/*.c"
)

target_sources(fsync_lib
  PRIVATE
    ${SOURCES}
)

target_include_directories(fsync_lib
  PUBLIC
    ${CMAKE_SOURCE_DIR}/hal/include
    ${CMAKE_SOURCE_DIR}/targets/${TARGET_PLATFORM}/include
    "include/"
)

target_link_libraries(fsync_lib
  PUBLIC
    runtime
)