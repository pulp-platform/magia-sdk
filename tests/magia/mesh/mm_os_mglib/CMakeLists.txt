# Copyright 2024-2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Moritz Scherer <scheremo@iis.ee.ethz.ch>
# Philip Wiese <wiesep@iis.ee.ethz.ch>
# Alberto Dequino <alberto.dequino@unibo.it>

set(TEST_NAME test_mm_os_mglib)

file(GLOB_RECURSE TEST_SRCS
  "src/*.c"
)

add_executable(${TEST_NAME} ${TEST_SRCS})
target_include_directories(${TEST_NAME} PUBLIC include)

target_compile_options(${TEST_NAME}
  PRIVATE
  -O3
  -flto
)
# LTO at link folds the mglib wrappers into this test's call sites.
# --allow-multiple-definition: io.c's libc stubs (memcpy/memset/...) are compiled
# both here and into libeu_lib.a; identical source, so taking the first is safe.
# (Pre-existing duplication that LTO's symbol resolver exposes; mirrors the Spatz
# link config in cmake/spatz_config.cmake.)
target_link_options(${TEST_NAME} PRIVATE -flto -Wl,--allow-multiple-definition)
target_link_libraries(${TEST_NAME} PUBLIC runtime mglib)

add_custom_command(
        TARGET ${TEST_NAME}
        POST_BUILD
        COMMAND ${CMAKE_OBJDUMP} -dhS -Mmarch=${ISA} $<TARGET_FILE:${TEST_NAME}> > $<TARGET_FILE:${TEST_NAME}>.s)
