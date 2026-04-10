set(TEST_NAME test_gemm_fifo)

set(FIFO_BATCH_ROWS "1" CACHE STRING "Rows per FIFO push (default 1)")

file(GLOB_RECURSE TEST_SRCS "src/*.c")

add_executable(${TEST_NAME} ${TEST_SRCS})
target_include_directories(${TEST_NAME} PUBLIC include ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_options(${TEST_NAME} PRIVATE -O2)
target_compile_definitions(${TEST_NAME} PRIVATE FIFO_BATCH_ROWS=${FIFO_BATCH_ROWS})
target_link_libraries(${TEST_NAME} PUBLIC runtime hal)

add_custom_command(
    TARGET ${TEST_NAME} POST_BUILD
    COMMAND ${CMAKE_OBJDUMP} -dhS -Mmarch=${ISA} $<TARGET_FILE:${TEST_NAME}> > $<TARGET_FILE:${TEST_NAME}>.s)
