set(TEST_NAME hello_spatz_pulp)

# Compile Spatz task (generates hello_spatz_pulp_task_bin.h)
add_spatz_task(
    TEST_NAME ${TEST_NAME}
    TASK_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/spatz_task/hello_task.c
    FIRST_TASK_NAME hello_task
)

# Compile PULP task (generates hello_spatz_pulp_bin.h)
add_pulp_task(
    TEST_NAME ${TEST_NAME}
    TASK_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/pulp_task/hello_task.c
)

# Compile CV32 executable embedding both binaries
add_cv32_executable_with_pulp_and_spatz(
    TARGET_NAME ${TEST_NAME}
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/main.c
)
