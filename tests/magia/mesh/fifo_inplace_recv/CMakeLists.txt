# Copyright 2026 Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

set(TEST_NAME test_fifo_inplace_recv)

add_executable(${TEST_NAME} src/test.c)
target_compile_options(${TEST_NAME} PRIVATE -O2)
target_link_libraries(${TEST_NAME} PUBLIC runtime hal)

# Same plan through the unpack path, for A/B comparison.
add_executable(${TEST_NAME}_unpack src/test.c)
target_compile_options(${TEST_NAME}_unpack PRIVATE -O2)
target_compile_definitions(${TEST_NAME}_unpack PRIVATE MAPS_FIFO_INPLACE_RECV=0)
target_link_libraries(${TEST_NAME}_unpack PUBLIC runtime hal)

# Blocking runner variants: no queued iDMA transfers.
foreach(VARIANT blocking blocking_unpack)
    add_executable(${TEST_NAME}_${VARIANT} src/test.c)
    target_compile_options(${TEST_NAME}_${VARIANT} PRIVATE -O2)
    target_compile_definitions(${TEST_NAME}_${VARIANT} PRIVATE FIFO_INPLACE_BLOCKING)
    target_link_libraries(${TEST_NAME}_${VARIANT} PUBLIC runtime hal)
endforeach()
target_compile_definitions(${TEST_NAME}_blocking_unpack PRIVATE MAPS_FIFO_INPLACE_RECV=0)
