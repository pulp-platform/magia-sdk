set(TEST_NAME hello_spatz)

# Compile Spatz task
add_spatz_task(
    TEST_NAME ${TEST_NAME}
    TASK_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/spatz_task/hello_task.c
    FIRST_TASK_NAME hello_task
    INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/tests/utils
)

# Compile CV32 executable with embedded Spatz binary
add_cv32_executable_with_spatz(
    TARGET_NAME ${TEST_NAME}
    SPATZ_HEADER ${SPATZ_HEADER}
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/main.c
    INCLUDE_DIRS
        ${CMAKE_SOURCE_DIR}/tests/utils
)
