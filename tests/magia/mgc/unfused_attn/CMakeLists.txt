# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

set(TEST_NAME test_unfused_attn)
set(MGC_KERNEL unfused_attn)
set(MGC_KERNEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../kernels/mgc_kernels/${MGC_KERNEL})

file(GLOB_RECURSE TEST_SRCS
  "src/*.c"
  "${MGC_KERNEL_DIR}/src/*.c"
)

add_executable(${TEST_NAME} ${TEST_SRCS})
target_include_directories(${TEST_NAME} PUBLIC include ${MGC_KERNEL_DIR}/include)

# The SDK Spatz kernels it uses, their tasks in one Spatz binary
add_spatz_kernels(
  TARGET ${TEST_NAME}
  TEST_NAME unfused_attn_spatz
  KERNELS softmax_rows
)

target_compile_options(${TEST_NAME}
  PRIVATE
  -O3
  -flto
)
# As in flat_attn_pipeopt: LTO folds the mglib wrappers into the call sites, and io.c's
# libc stubs are compiled both here and into libeu_lib.a.
target_link_options(${TEST_NAME} PRIVATE -flto -Wl,--allow-multiple-definition)
target_link_libraries(${TEST_NAME} PUBLIC runtime mglib)

add_custom_command(
        TARGET ${TEST_NAME}
        POST_BUILD
        COMMAND ${CMAKE_OBJDUMP} -dhS -Mmarch=${ISA} $<TARGET_FILE:${TEST_NAME}> > $<TARGET_FILE:${TEST_NAME}>.s)
