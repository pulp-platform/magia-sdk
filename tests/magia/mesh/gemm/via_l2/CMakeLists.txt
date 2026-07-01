# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

set(TEST_NAME test_gemm)

set(GEMM_TEST_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/test.h)
set(GEN_SCRIPT    ${CMAKE_CURRENT_SOURCE_DIR}/gen_golden.py)

add_custom_command(
    OUTPUT  ${GEMM_TEST_HEADER}
    COMMAND python3 ${GEN_SCRIPT} --output ${GEMM_TEST_HEADER}
    DEPENDS ${GEN_SCRIPT}
    COMMENT "Generating GEMM chain golden data (test.h)"
    VERBATIM)

file(GLOB_RECURSE TEST_SRCS "src/*.c")

add_executable(${TEST_NAME} ${TEST_SRCS} ${GEMM_TEST_HEADER})
target_include_directories(${TEST_NAME} PUBLIC include)
target_compile_options(${TEST_NAME} PRIVATE -O2)
target_link_libraries(${TEST_NAME} PUBLIC runtime hal)

add_custom_command(
    TARGET ${TEST_NAME} POST_BUILD
    COMMAND ${CMAKE_OBJDUMP} -dhS -Mmarch=${ISA} $<TARGET_FILE:${TEST_NAME}> > $<TARGET_FILE:${TEST_NAME}>.s)
