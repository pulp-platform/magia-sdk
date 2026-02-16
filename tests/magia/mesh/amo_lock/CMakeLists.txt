# Copyright 2024-2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Moritz Scherer <scheremo@iis.ee.ethz.ch>
# Philip Wiese <wiesep@iis.ee.ethz.ch>
# Alberto Dequino <alberto.dequino@unibo.it>

set(TEST_NAME test_amo_lock)

file(GLOB_RECURSE TEST_SRCS
  "src/*.c"
)

add_executable(${TEST_NAME} ${TEST_SRCS})
target_include_directories(${TEST_NAME} PUBLIC include)

target_compile_options(${TEST_NAME}
  PRIVATE
  -O2
)
target_link_libraries(${TEST_NAME} PUBLIC runtime hal)

add_custom_command(
        TARGET ${TEST_NAME}
        POST_BUILD
        COMMAND ${CMAKE_OBJDUMP} -dhS $<TARGET_FILE:${TEST_NAME}> > $<TARGET_FILE:${TEST_NAME}>.s)