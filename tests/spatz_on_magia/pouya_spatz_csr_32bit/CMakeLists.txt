cmake_minimum_required(VERSION 3.18)

# TEST_NAME -> generates <TEST_NAME>_task_bin.h (defines SPATZ_CSR_TASK,
# SPATZ_BINARY_START), which main.c includes.
set(TEST_NAME spatz_csr)

# --- Spatz kernel (Snitch + Spatz) ------------------------------------
# NOTE: per the README naming rule (guide, section 3.2), the task source
# should be named "<name>_task.c" (here: spatz_csr_task.c). This project's
# task file is delivered as task.c to match your requested file list --
# place/rename it accordingly (e.g. spatz_csr_task.c, optionally under a
# spatz_task/ subdirectory) or adjust TASK_SOURCES below to match.
add_spatz_task(
    TEST_NAME       ${TEST_NAME}
    TASK_SOURCES    ${CMAKE_CURRENT_SOURCE_DIR}/spatz_task/onnx_spmv_task.c
    FIRST_TASK_NAME spatz_csr_task
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/test_data
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# --- CV32 host executable, with the Spatz binary embedded -------------
add_cv32_executable_with_spatz(
    TARGET_NAME  ${TEST_NAME}
    SOURCES      ${CMAKE_CURRENT_SOURCE_DIR}/main.c
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/test_data
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# data.h (CSR data + golden) is expected in the same include path as
# params.h/mem_layout.h above -- drop it into this directory, or add its
# location to both INCLUDE_DIRS lists if you keep it elsewhere
# (e.g. a test_data/ subfolder), per the guide's reference layout.