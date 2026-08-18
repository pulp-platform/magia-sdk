# PULP-only variant of multi_task (no Spatz), to isolate the PULP path.
# Only built in PULP-only configurations (pulp_on_magia is excluded when both
# SPATZ_TESTS and PULP_TESTS are on — see tests/CMakeLists.txt), so this target
# name does not clash with spatz_pulp_on_magia/multi_task.
set(TEST_NAME multi_task)

# Compile PULP tasks: vec_sum + vec_dot + vec_scale + vec_absmax → multi_task_bin.h
add_pulp_task(
    TEST_NAME ${TEST_NAME}
    TASK_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/pulp_task/vec_sum_pulp_task.c
        ${CMAKE_CURRENT_SOURCE_DIR}/pulp_task/vec_dot_pulp_task.c
        ${CMAKE_CURRENT_SOURCE_DIR}/pulp_task/vec_scale_pulp_task.c
        ${CMAKE_CURRENT_SOURCE_DIR}/pulp_task/vec_absmax_pulp_task.c
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Compile CV32 executable embedding only the PULP binary
add_cv32_executable_with_pulp(
    TARGET_NAME  ${TEST_NAME}
    SOURCES      ${CMAKE_CURRENT_SOURCE_DIR}/main.c
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
)
