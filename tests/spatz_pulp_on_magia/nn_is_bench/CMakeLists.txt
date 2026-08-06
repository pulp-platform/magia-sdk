# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

set(TEST_NAME nn_is_bench)

# ==============================================================================
# Dataset
#
# include/nn_is_data.h and include/nn_is_size.h are GENERATED, not committed: at
# the default geometry the data header alone is 6.3 MB of hex literals, and every
# change of size or timeslot count would park another copy of it in git history
# forever. The generator is deterministic from its seed, so the bytes are fully
# reproducible from the four numbers below.
#
# Geometry knobs, settable from the cmake command line:
#   -DNN_SIZE=512 -DNN_TIMESLOTS=16 -DNN_SEED=0xC0FFEE
# The mesh side comes from TILES, which the top-level Makefile already passes.
#
# Requires python3 with numpy at build time.
# ==============================================================================

set(NN_SIZE      512      CACHE STRING "nn_is_bench: M=N=K of the GEMM")
set(NN_TIMESLOTS 16       CACHE STRING "nn_is_bench: K timeslots (>= TILES^2 to pipeline well)")
set(NN_SEED      0xC0FFEE CACHE STRING "nn_is_bench: dataset RNG seed")

set(NN_GEN_SCRIPT ${CMAKE_SOURCE_DIR}/scripts/gen_nn_is_data.py)
set(NN_GEN_DIR    ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(NN_GEN_DATA   ${NN_GEN_DIR}/nn_is_data.h)
set(NN_GEN_SIZE   ${NN_GEN_DIR}/nn_is_size.h)

# Custom commands do not notice when only their command line changes, so the
# geometry is mirrored into a stamp file that is rewritten only when it actually
# differs. Depending on the stamp is what makes -DNN_SIZE=... retrigger.
set(NN_GEN_PARAMS "${NN_SIZE} ${NN_TIMESLOTS} ${NN_SEED} ${TILES}")
set(NN_GEN_STAMP  ${CMAKE_CURRENT_BINARY_DIR}/nn_is_gen.params)
if(EXISTS ${NN_GEN_STAMP})
    file(READ ${NN_GEN_STAMP} NN_GEN_PARAMS_OLD)
else()
    set(NN_GEN_PARAMS_OLD "")
endif()
if(NOT NN_GEN_PARAMS_OLD STREQUAL NN_GEN_PARAMS)
    file(WRITE ${NN_GEN_STAMP} "${NN_GEN_PARAMS}")
endif()

add_custom_command(
    OUTPUT ${NN_GEN_DATA} ${NN_GEN_SIZE}
    COMMAND python3 ${NN_GEN_SCRIPT}
            --size ${NN_SIZE}
            --timeslots ${NN_TIMESLOTS}
            --seed ${NN_SEED}
            --mesh ${TILES}
            -o ${NN_GEN_DIR}
    DEPENDS ${NN_GEN_SCRIPT} ${NN_GEN_STAMP}
    COMMENT "[nn_is_bench] Generating dataset ${NN_SIZE}^3, ${NN_TIMESLOTS} timeslots, ${TILES}x${TILES} mesh..."
    VERBATIM
)
add_custom_target(nn_is_bench_dataset DEPENDS ${NN_GEN_DATA} ${NN_GEN_SIZE})

# ==============================================================================
# Accelerator binaries and the CV32 executable
# ==============================================================================

# Spatz: fp16 activation + row-wise LayerNorm -> nn_spatz_task_bin.h
add_spatz_task(
    TEST_NAME       nn_spatz
    FIRST_TASK_NAME nn_act_spatz_task
    TASK_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/spatz_task/nn_act_spatz_task.c
        ${CMAKE_CURRENT_SOURCE_DIR}/spatz_task/nn_ln_spatz_task.c
    INCLUDE_DIRS
        ${NN_GEN_DIR}
)

# PULP cluster: int16 packed-SIMD row-wise softmax/argmax -> nn_pulp_task_bin.h
add_pulp_task(
    TEST_NAME nn_pulp
    TASK_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/pulp_task/nn_softmax_pulp_task.c
    INCLUDE_DIRS
        ${NN_GEN_DIR}
)

# CV32 control core, embedding both accelerator binaries
add_cv32_executable_with_pulp_and_spatz(
    TARGET_NAME     ${TEST_NAME}
    SPATZ_TEST_NAME nn_spatz
    PULP_TEST_NAME  nn_pulp
    SOURCES         ${CMAKE_CURRENT_SOURCE_DIR}/main.c
    INCLUDE_DIRS    ${NN_GEN_DIR}
)

# All three consume nn_is_size.h (via nn_is_params.h) and main.c also consumes
# nn_is_data.h, so all three have to wait for the generator. The task helpers do
# their work inside their own targets, whose names they export.
add_dependencies(${nn_spatz_SPATZ_TARGET} nn_is_bench_dataset)
add_dependencies(${nn_pulp_PULP_TARGET}   nn_is_bench_dataset)
add_dependencies(${TEST_NAME}             nn_is_bench_dataset)

# The device-side golden check is ~36k CV32 cycles sitting between GEMM rounds,
# where it plausibly costs more than itself by holding the tile back from the next
# round. It is excluded from cyc_total; the sync time it induces is not. Turning it
# off is how you measure the kernel alone:
#   cmake -DNN_VERIFY=OFF -B build     (the cache setting then sticks)
option(NN_VERIFY "nn_is_bench: run the device-side golden check" ON)
if(NOT NN_VERIFY)
    target_compile_definitions(${TEST_NAME} PRIVATE NN_STANDALONE_FLAGS=0)
    message(STATUS "[nn_is_bench] golden check DISABLED")
endif()
