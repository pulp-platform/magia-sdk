set(TEST_NAME onnx_div)

# Compile Spatz task
add_spatz_task(
    TEST_NAME ${TEST_NAME}
    TASK_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/spatz_task/onnx_div_task.c
    FIRST_TASK_NAME onnx_div_task
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/test_data
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Compile CV32 executable with embedded Spatz binary
add_cv32_executable_with_spatz(
    TARGET_NAME ${TEST_NAME}
    SPATZ_HEADER ${SPATZ_HEADER}
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/main.c
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/test_data
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
