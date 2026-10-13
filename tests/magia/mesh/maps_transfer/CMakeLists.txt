set(TEST_NAME test_maps_transfer)

add_executable(${TEST_NAME} src/test.c)
target_compile_options(${TEST_NAME} PRIVATE -O2)
target_link_libraries(${TEST_NAME} PUBLIC runtime hal)
