# Copyright 2024-2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Moritz Scherer <scheremo@iis.ee.ethz.ch>
# Viviane Potocnik <vivianep@iis.ee.ethz.ch>
# Alberto Dequino <alberto.dequino@unibo.it>

set(AVAILABLE_TARGETS
  "magia_v1"
  "magia_v2"
  CACHE STRING "Available Targets"
)

get_property(OPT_STRINGS CACHE OPT PROPERTY STRINGS)

if(NOT TARGET_PLATFORM IN_LIST AVAILABLE_TARGETS)
  message(FATAL_ERROR "Wrong value for TARGET_PLATFORM: Got ${TARGET_PLATFORM}. Available targets are: ${AVAILABLE_TARGETS}")
endif()

message(STATUS "[MAGIA-SDK] TARGET_PLATFORM       : ${TARGET_PLATFORM}")

add_subdirectory(${TARGET_PLATFORM})