set(TEST_NAME test_fork)

add_pulp_task(
    TEST_NAME ${TEST_NAME}
    TASK_SOURCES 
        ${CMAKE_CURRENT_SOURCE_DIR}/pulp_task/fork_task.c
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_cv32_executable_with_pulp(
    TARGET_NAME ${TEST_NAME}
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/main.c
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
)
