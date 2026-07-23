set(TEST_NAME hello_pulp)

add_pulp_task(
    TEST_NAME ${TEST_NAME}
    TASK_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/pulp_task/hello_task.c
)

add_cv32_executable_with_pulp(
    TARGET_NAME ${TEST_NAME}
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/main.c
)
