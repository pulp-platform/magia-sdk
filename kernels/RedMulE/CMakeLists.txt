# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# RedMulE-backed kernels (extracted from the maps test)

add_library(kernels_redmule STATIC)

file(GLOB_RECURSE SOURCES
  "src/*.c"
)

target_sources(kernels_redmule
  PRIVATE
    ${SOURCES}
)

target_include_directories(kernels_redmule
  PUBLIC
    ${CMAKE_SOURCE_DIR}/hal/include
    ${CMAKE_SOURCE_DIR}/targets/${TARGET_PLATFORM}/include
    "include/"
)

target_link_libraries(kernels_redmule
  PUBLIC
    runtime
)
