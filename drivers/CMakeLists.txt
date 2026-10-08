# Copyright 2024-2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Viviane Potocnik <vivianep@iis.ee.ethz.ch>
# Philip Wiese <wiesep@iis.ee.ethz.ch>
# Alberto Dequino <alberto.dequino@unibo.it>

# Define mappings for drivers
set(DRIVER_MAPPINGS
    magia:idma32,redmule16,fsync32,eventunit32
)

# Call the macro
add_magia_subdirectories(${TARGET_PLATFORM} "Driver" "${DRIVER_MAPPINGS}")

target_include_directories(runtime PUBLIC ${CMAKE_CURRENT_LIST_DIR})