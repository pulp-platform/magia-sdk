# Full-mesh reference for the MobileViT nodes 170 through 179 paper slice.
set(TEST_NAME onnx_mobilevit_slice)
set(KERNELS_DIR ${CMAKE_SOURCE_DIR}/kernels/spatz_fp16)
set(SLICE_KERNELS groupnorm conv2dgemm softmax mul reducesum relu add)

if(NOT MOBILEVIT_SLICE_TOKENS)
    message(FATAL_ERROR "MOBILEVIT_SLICE_TOKENS is required")
endif()

set(SLICE_TASK_SOURCES "")
set(SLICE_KERNEL_SOURCES "")
set(SLICE_KERNEL_INCLUDES "")
foreach(K ${SLICE_KERNELS})
    list(APPEND SLICE_TASK_SOURCES ${KERNELS_DIR}/${K}/spatz_task/${K}_fp16_spatz_task.c)
    list(APPEND SLICE_KERNEL_SOURCES ${KERNELS_DIR}/${K}/src/${K}_fp16_spatz.c)
    list(APPEND SLICE_KERNEL_INCLUDES ${KERNELS_DIR}/${K}/include)
endforeach()
list(APPEND SLICE_TASK_SOURCES
    ${KERNELS_DIR}/mul/spatz_task/mul_bcast_fp16_spatz_task.c
    ${KERNELS_DIR}/add/spatz_task/add_bcast_fp16_spatz_task.c
)

add_spatz_task(
    TEST_NAME ${TEST_NAME}
    TASK_SOURCES ${SLICE_TASK_SOURCES}
    FIRST_TASK_NAME groupnorm_fp16_spatz_task
    INCLUDE_DIRS ${SLICE_KERNEL_INCLUDES}
)

configure_file(slice_data.S.in ${CMAKE_CURRENT_BINARY_DIR}/slice_data.S @ONLY)
set_property(SOURCE ${CMAKE_CURRENT_BINARY_DIR}/slice_data.S
    APPEND PROPERTY OBJECT_DEPENDS ${MOBILEVIT_SLICE_DATA})
add_cv32_executable_with_spatz(
    TARGET_NAME ${TEST_NAME}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/main.c
        ${CMAKE_CURRENT_BINARY_DIR}/slice_data.S
        ${SLICE_KERNEL_SOURCES}
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/shim
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${SLICE_KERNEL_INCLUDES}
)
target_compile_definitions(${TEST_NAME} PRIVATE
    MOBILEVIT_SLICE_TOKENS=${MOBILEVIT_SLICE_TOKENS}
)
target_compile_options(${TEST_NAME} PRIVATE -O2)
