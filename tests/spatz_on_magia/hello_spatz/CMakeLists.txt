# Compile Spatz task
add_spatz_task(
    TEST_NAME hello_spatz
    CRT0_SRC ${SPATZ_CRT0_SRC}
    TASK_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/spatz_task/hello.c
    LINKER_SCRIPT ${SPATZ_LINK_SCRIPT}
    OUTPUT_DIR ${CMAKE_BINARY_DIR}
    OUTPUT_SPATZ_DIR ${CMAKE_BINARY_DIR}/bin/spatz/hello_spatz
    OUTPUT_VAR HELLO_SPATZ_HEADER
    FIRST_TASK_NAME hello_world_simple_task
    INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/tests/utils
)

# Compile CV32 executable with embedded Spatz binary
add_cv32_executable_with_spatz(
    TARGET_NAME hello_spatz
    SPATZ_HEADER ${HELLO_SPATZ_HEADER}
    CRT0_SRC ${CV32_CRT0_SRC}
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/main.c
    LINK_SCRIPT ${CV32_LINK_SCRIPT}
    INCLUDE_DIRS
        ${CMAKE_SOURCE_DIR}/tests/utils
)
