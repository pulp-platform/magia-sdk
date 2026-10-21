# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

set(TEST_NAME test_unfused_attn)

set(KERNELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../kernels)
set(SOFTMAX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kernels/softmax_rows/fp16/spatz)
set(SPATZ_KERNEL_INCS ${KERNELS_DIR}/common ${SOFTMAX_DIR}/include)

# The row softmax kernel -> unfused_attn_spatz_task_bin.h
add_spatz_task(
  TEST_NAME unfused_attn_spatz
  TASK_SOURCES ${SOFTMAX_DIR}/spatz_task/softmax_rows_fp16_spatz_task.c
  FIRST_TASK_NAME softmax_rows_fp16_spatz_task
  INCLUDE_DIRS ${SPATZ_KERNEL_INCS}
)
get_filename_component(SPATZ_HEADER_DIR ${unfused_attn_spatz_SPATZ_HEADER} DIRECTORY)

file(GLOB_RECURSE TEST_SRCS
  "src/*.c"
)

add_executable(${TEST_NAME} ${TEST_SRCS})
add_dependencies(${TEST_NAME} ${unfused_attn_spatz_SPATZ_TARGET})
# Spatz boots from spatz_init.bin (as in add_cv32_executable_with_spatz)
if(TARGET spatz_bootrom)
  add_dependencies(${TEST_NAME} spatz_bootrom)
endif()
target_include_directories(${TEST_NAME} PUBLIC include)
target_include_directories(${TEST_NAME} PRIVATE ${SPATZ_KERNEL_INCS} ${SPATZ_HEADER_DIR})

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
