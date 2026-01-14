# Copyright 2024-2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Moritz Scherer <scheremo@iis.ee.ethz.ch>
# Viviane Potocnik <vivianep@iis.ee.ethz.ch>
# Alberto Dequino <alberto.dequino@unibo.it>

file(GLOB_RECURSE SOURCES
  "src/*"
)

add_library(hal STATIC ${SOURCES})

target_include_directories(hal
  PUBLIC
  "include"
)

if(${TARGET_PLATFORM} STREQUAL "magia_v1")
target_link_libraries(hal
  PUBLIC
    "idma_lib"
    "redmule_lib"
    "fsync_lib"
)
elseif(${TARGET_PLATFORM} STREQUAL "magia_v2")
target_link_libraries(hal
  PUBLIC
    "idma_lib"
    "redmule_lib"
    "fsync_lib"
    "eu_lib"
)
endif()