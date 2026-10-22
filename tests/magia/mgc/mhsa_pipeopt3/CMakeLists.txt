# Copyright 2024-2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Moritz Scherer <scheremo@iis.ee.ethz.ch>
# Philip Wiese <wiesep@iis.ee.ethz.ch>
# Alberto Dequino <alberto.dequino@unibo.it>

set(TEST_NAME test_mhsa_pipeopt3)

set(KERNELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../kernels)
set(LOCAL_KERNELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kernels)
# Generic kernels from kernels/, merged softmax kernels from this test's kernels/
set(SPATZ_KERNELS add rowdiv rowscale)
set(LOCAL_SPATZ_KERNELS colmax colsum_rescale expminus_rowsum rowmax)

# All the Spatz kernels in one task binary -> mhsa_pipeopt3_spatz_task_bin.h
set(SPATZ_TASK_SRCS)
set(SPATZ_KERNEL_SRCS)
set(SPATZ_KERNEL_INCS ${KERNELS_DIR}/common)
foreach(K ${SPATZ_KERNELS})
  list(APPEND SPATZ_TASK_SRCS ${KERNELS_DIR}/${K}/fp16/spatz/spatz_task/${K}_fp16_spatz_task.c)
  list(APPEND SPATZ_KERNEL_SRCS ${KERNELS_DIR}/${K}/fp16/spatz/src/${K}_fp16_spatz.c)
  list(APPEND SPATZ_KERNEL_INCS ${KERNELS_DIR}/${K}/fp16/spatz/include)
endforeach()
foreach(K ${LOCAL_SPATZ_KERNELS})
  list(APPEND SPATZ_TASK_SRCS ${LOCAL_KERNELS_DIR}/${K}/fp16/spatz/spatz_task/${K}_fp16_spatz_task.c)
  list(APPEND SPATZ_KERNEL_SRCS ${LOCAL_KERNELS_DIR}/${K}/fp16/spatz/src/${K}_fp16_spatz.c)
  list(APPEND SPATZ_KERNEL_INCS ${LOCAL_KERNELS_DIR}/${K}/fp16/spatz/include)
endforeach()
add_spatz_task(
  TEST_NAME mhsa_pipeopt3_spatz
  TASK_SOURCES ${SPATZ_TASK_SRCS}
  FIRST_TASK_NAME add_fp16_spatz_task
  INCLUDE_DIRS ${SPATZ_KERNEL_INCS}
  # Run on every K/V step, in this order
  HOT_TASKS rowscale_fp16_spatz_task rowmax_fp16_spatz_task colmax_fp16_spatz_task
            expminus_rowsum_fp16_spatz_task colsum_rescale_fp16_spatz_task
)

# Each kernel source includes its own <kernel>_task_bin.h: point them all to the shared header,
# without redefining the binary array (defined once, in src/kernels.c).
set(SPATZ_SHIM_DIR ${CMAKE_CURRENT_BINARY_DIR}/spatz_shim)
foreach(K ${SPATZ_KERNELS} ${LOCAL_SPATZ_KERNELS})
  file(WRITE ${SPATZ_SHIM_DIR}/${K}_fp16_spatz_task_bin.h
       "#define SPATZ_BINARY_NO_DEFINE\n#include \"mhsa_pipeopt3_spatz_task_bin.h\"\n")
endforeach()
get_filename_component(SPATZ_HEADER_DIR ${mhsa_pipeopt3_spatz_SPATZ_HEADER} DIRECTORY)

file(GLOB_RECURSE TEST_SRCS
  "src/*.c"
)

add_executable(${TEST_NAME} ${TEST_SRCS} ${SPATZ_KERNEL_SRCS})
add_dependencies(${TEST_NAME} ${mhsa_pipeopt3_spatz_SPATZ_TARGET})
# Spatz boots from spatz_init.bin (as in add_cv32_executable_with_spatz)
if(TARGET spatz_bootrom)
  add_dependencies(${TEST_NAME} spatz_bootrom)
endif()
target_include_directories(${TEST_NAME} PUBLIC include)
target_include_directories(${TEST_NAME} PRIVATE ${SPATZ_KERNEL_INCS} ${SPATZ_HEADER_DIR})
set_source_files_properties(${SPATZ_KERNEL_SRCS} PROPERTIES INCLUDE_DIRECTORIES ${SPATZ_SHIM_DIR})

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
