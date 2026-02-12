# Compile Spatz task
add_spatz_task(
    TEST_NAME onnx_add
    CRT0_SRC ${SPATZ_CRT0_SRC}
    TASK_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/spatz_task/onnx_add_task.c
    LINKER_SCRIPT ${SPATZ_LINK_SCRIPT}
    OUTPUT_DIR ${CMAKE_BINARY_DIR}
    OUTPUT_SPATZ_DIR ${CMAKE_BINARY_DIR}/bin/spatz/onnx_add
    OUTPUT_VAR ONNX_ADD_HEADER
    FIRST_TASK_NAME onnx_add_task
    INCLUDE_DIRS
        ${CMAKE_SOURCE_DIR}/tests/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/test_data
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Compile CV32 executable with embedded Spatz binary
add_cv32_executable_with_spatz(
    TARGET_NAME onnx_add
    SPATZ_HEADER ${ONNX_ADD_HEADER}
    CRT0_SRC ${CV32_CRT0_SRC}
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/main.c
    LINK_SCRIPT ${CV32_LINK_SCRIPT}
    INCLUDE_DIRS
        ${CMAKE_SOURCE_DIR}/tests/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/test_data
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
