# Copyright 2024-2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Moritz Scherer <scheremo@iis.ee.ethz.ch>
# Philip Wiese <wiesep@iis.ee.ethz.ch>
# Alberto Dequino <alberto.dequino@unibo.it>

set(TEST_NAME test_flat_attn_pipeopt)
set(MGC_KERNEL flat_attn_pipeopt)
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
  TEST_NAME flat_attn_pipeopt_spatz
  KERNELS add memset rowdiv rowscale colmax colsum_rescale expminus_rowsum rowmax
  # Run on every K/V step, in this order
  HOT_TASKS rowscale_fp16_spatz_task rowmax_fp16_spatz_task colmax_fp16_spatz_task expminus_rowsum_fp16_spatz_task colsum_rescale_fp16_spatz_task
)

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
