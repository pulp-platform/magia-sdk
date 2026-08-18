set(TEST_NAME multi_task)

# Compile Spatz tasks: vec_add + vec_relu → multi_spatz_task_bin.h
add_spatz_task(
    TEST_NAME       multi_spatz
    FIRST_TASK_NAME vec_add_spatz_task
    TASK_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/spatz_task/vec_add_spatz_task.c
        ${CMAKE_CURRENT_SOURCE_DIR}/spatz_task/vec_relu_spatz_task.c
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Compile PULP tasks: vec_sum + vec_dot + vec_scale + vec_absmax → multi_pulp_task_bin.h
add_pulp_task(
    TEST_NAME multi_pulp
    TASK_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/pulp_task/vec_sum_pulp_task.c
        ${CMAKE_CURRENT_SOURCE_DIR}/pulp_task/vec_dot_pulp_task.c
        ${CMAKE_CURRENT_SOURCE_DIR}/pulp_task/vec_scale_pulp_task.c
        ${CMAKE_CURRENT_SOURCE_DIR}/pulp_task/vec_absmax_pulp_task.c
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Compile CV32 executable embedding both binaries
add_cv32_executable_with_pulp_and_spatz(
    TARGET_NAME     ${TEST_NAME}
    SPATZ_TEST_NAME multi_spatz
    PULP_TEST_NAME  multi_pulp
    SOURCES         ${CMAKE_CURRENT_SOURCE_DIR}/main.c
    INCLUDE_DIRS    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
