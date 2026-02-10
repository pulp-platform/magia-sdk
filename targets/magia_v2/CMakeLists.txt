# Copyright 2024-2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Moritz Scherer <scheremo@iis.ee.ethz.ch>
# Viviane Potocnik <vivianep@iis.ee.ethz.ch>
# Philip Wiese <wiesep@iis.ee.ethz.ch>
# Alberto Dequino <alberto.dequino@unibo.it>

add_library(runtime STATIC)

file(GLOB_RECURSE ASM_SOURCES
  "src/crt0.S"
)

file(GLOB_RECURSE C_SOURCES
  "src/*.c"
)

set_property(SOURCE ${ASM_SOURCES} PROPERTY LANGUAGE ASM)

target_sources(runtime
  PUBLIC
  ${ASM_SOURCES}
  ${C_SOURCES}
)

target_include_directories(runtime
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/include
)

target_compile_options(runtime
  PUBLIC
  -O3
  -DEVAL=${EVAL}
  -DFSYNC_MM=${FSYNC_MM}
  -DIDMA_MM=${IDMA_MM}
  -DREDMULE_MM=${REDMULE_MM}
  -DSTALLING=${STALLING}
  -DPROFILE_CMP=${PROFILE_CMP}
  -DPROFILE_CMI=${PROFILE_CMI}
  -DPROFILE_CMO=${PROFILE_CMO}
  -DPROFILE_SNC=${PROFILE_SNC}
)

target_compile_options(runtime
  PUBLIC
  -march=${ISA}
  -mabi=${ABI}
  -D__riscv__
  -fdata-sections -ffunction-sections #ZL-MOD 03072025
  -Wno-incompatible-pointer-types #ZL-MOD 03072025
  -Wno-int-conversion #ZL-MOD 03072025
)

target_link_options(runtime
  PUBLIC
  -L${CMAKE_CURRENT_LIST_DIR}
  -Tlink.ld
  -march=${ISA}
  -mabi=${ABI}
  -nostartfiles
  -nostdlib
  -ffreestanding
  -Wno-incompatible-pointer-types #ZL-MOD 03072025
  -Wno-int-conversion #ZL-MOD 03072025
  -Wl,--gc-sections #ZL-MOD 03072025
)


# ==================== SPATZ INTEGRATION ====================

# Include Spatz configuration and helpers if Spatz is available
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/spatz")
    include(${CMAKE_SOURCE_DIR}/cmake/spatz_config.cmake)
    include(${CMAKE_SOURCE_DIR}/cmake/spatz_helpers.cmake)
    include(${CMAKE_SOURCE_DIR}/cmake/bootrom_helpers.cmake)

    # Build bootrom only if Spatz tests are enabled
    if(SPATZ_TESTS)
        add_spatz_bootrom(
            BOOTROM_SRC ${CMAKE_CURRENT_LIST_DIR}/spatz/bootrom/spatz_init.S
            LINKER_SCRIPT ${CMAKE_CURRENT_LIST_DIR}/spatz/bootrom/spatz_init.ld
            OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/bootrom
        )
        message(STATUS "[MAGIA-SDK] Spatz integration enabled for magia_v2 (bootrom will be built)")
    else()
        message(STATUS "[MAGIA-SDK] Spatz integration disabled (SPATZ_TESTS=0)")
    endif()
endif()

# =========================================================
