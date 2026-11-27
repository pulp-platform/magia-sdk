# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
#
# One test per Spatz kernel <k> (directory <k>/, target test_spatz_l1_<k>): it runs the full API
# MAGIA_<k>_fp16_spatz on L2 operands and the L1 API MAGIA_<k>_fp16_spatz_l1 on L1 copies of the
# same operands on every tile, and compares the two results.

set(KERNELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../kernels)

function(spatz_l1_test K)
  set(TEST_NAME test_spatz_l1_${K})
  set(KD ${KERNELS_DIR}/${K}/fp16/spatz)
  set(INCS ${KERNELS_DIR}/common ${KD}/include ${CMAKE_CURRENT_SOURCE_DIR}/common)

  # The kernel's task -> <k>_fp16_spatz_task_bin.h, included by the kernel source
  add_spatz_task(
    TEST_NAME ${K}_fp16_spatz
    TASK_SOURCES ${KD}/spatz_task/${K}_fp16_spatz_task.c
    FIRST_TASK_NAME ${K}_fp16_spatz_task
    INCLUDE_DIRS ${KERNELS_DIR}/common ${KD}/include
  )
  get_filename_component(SPATZ_HEADER_DIR ${${K}_fp16_spatz_SPATZ_HEADER} DIRECTORY)

  add_executable(${TEST_NAME} ${K}/src/test.c ${KD}/src/${K}_fp16_spatz.c)
  add_dependencies(${TEST_NAME} ${${K}_fp16_spatz_SPATZ_TARGET})
  if(TARGET spatz_bootrom)
    add_dependencies(${TEST_NAME} spatz_bootrom)
  endif()
  target_include_directories(${TEST_NAME} PRIVATE ${INCS} ${SPATZ_HEADER_DIR})
  target_compile_options(${TEST_NAME} PRIVATE -O2)
  target_link_options(${TEST_NAME} PRIVATE -Wl,--allow-multiple-definition)
  target_link_libraries(${TEST_NAME} PUBLIC runtime hal mglib)
  add_custom_command(
    TARGET ${TEST_NAME}
    POST_BUILD
    COMMAND ${CMAKE_OBJDUMP} -dhS -Mmarch=${ISA} $<TARGET_FILE:${TEST_NAME}> > $<TARGET_FILE:${TEST_NAME}>.s)
endfunction()

file(GLOB TESTS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*/src/test.c)
foreach(T ${TESTS})
  string(REGEX REPLACE "/src/test.c$" "" K ${T})
  spatz_l1_test(${K})
endforeach()
