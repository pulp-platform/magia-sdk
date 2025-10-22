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
  -ffreestanding
  -Wno-incompatible-pointer-types #ZL-MOD 03072025
  -Wno-int-conversion #ZL-MOD 03072025
  -Wl,--gc-sections #ZL-MOD 03072025
)