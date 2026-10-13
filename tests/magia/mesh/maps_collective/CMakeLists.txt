set(TEST_NAME test_maps_collective)

set(SPATZ_KERNELS_DIR ${CMAKE_SOURCE_DIR}/kernels)
add_spatz_task(
    TEST_NAME ${TEST_NAME}
    TASK_SOURCES
        ${SPATZ_KERNELS_DIR}/reducesum/fp16/spatz/spatz_task/reducesum_fp16_spatz_task.c
        ${SPATZ_KERNELS_DIR}/reducemax/fp16/spatz/spatz_task/reducemax_fp16_spatz_task.c
    FIRST_TASK_NAME reducesum_fp16_spatz_task
    INCLUDE_DIRS
        ${SPATZ_KERNELS_DIR}/reducesum/fp16/spatz/include
        ${SPATZ_KERNELS_DIR}/reducemax/fp16/spatz/include
)

add_cv32_executable_with_spatz(
    TARGET_NAME ${TEST_NAME}
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/test.c
)
target_link_libraries(${TEST_NAME} PUBLIC idma_lib fsync_lib eu_lib)
target_compile_definitions(${TEST_NAME} PRIVATE
    MAPS_HAS_REDUCE_SUM_SPATZ_TASK=1
    MAPS_HAS_REDUCE_MAX_SPATZ_TASK=1
)
