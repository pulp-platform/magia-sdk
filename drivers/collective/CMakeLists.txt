# Copyright 2024-2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Viviane Potocnik <vivianep@iis.ee.ethz.ch>
# Alberto Dequino <alberto.dequino@unibo.it>

# 32-bit Collective Driver CMake Configuration

add_library(collective_lib STATIC)

file(GLOB_RECURSE SOURCES
  "src/*.c"
)

target_sources(collective_lib
  PRIVATE
    ${SOURCES}
)

target_include_directories(collective_lib
  PUBLIC
    ${CMAKE_SOURCE_DIR}/hal/include
    ${CMAKE_SOURCE_DIR}/targets/${TARGET_PLATFORM}/include
    "include/"
)

target_link_libraries(collective_lib
  PUBLIC
    runtime
)