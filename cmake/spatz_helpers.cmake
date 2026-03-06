# Helpers for building Spatz tasks and CV32 executables that embed them.

# add_spatz_task(
#   TEST_NAME <name>
#   TASK_SOURCES <src...>
#   [FIRST_TASK_NAME <entry_symbol>]
#   [INCLUDE_DIRS <dirs...>]
# )
# Builds a Spatz task ELF/BIN/DUMP, generates <name>_task_bin.h and exports
# per-test variables (<TEST_NAME>_SPATZ_HEADER, <TEST_NAME>_TASK_BIN, ...).
function(add_spatz_task)
    set(options)
    set(oneValueArgs TEST_NAME FIRST_TASK_NAME)
    set(multiValueArgs TASK_SOURCES INCLUDE_DIRS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_TEST_NAME)
        message(FATAL_ERROR "add_spatz_task requires TEST_NAME")
    endif()
    if(NOT ARG_TASK_SOURCES)
        message(FATAL_ERROR "add_spatz_task(${ARG_TEST_NAME}) requires TASK_SOURCES")
    endif()
    if(NOT ARG_FIRST_TASK_NAME)
        set(ARG_FIRST_TASK_NAME "${ARG_TEST_NAME}_task")
    endif()

    # --------------------------------------------------------------------------
    # Paths and names
    # --------------------------------------------------------------------------
    set(SPATZ_OUTPUT_DIR "${SPATZ_TASK_OUTPUT_ROOT}/${ARG_TEST_NAME}/spatz_task")
    file(MAKE_DIRECTORY "${SPATZ_OUTPUT_DIR}")

    string(REGEX MATCH "_task$" HAS_TASK_SUFFIX "${ARG_TEST_NAME}")
    if(HAS_TASK_SUFFIX)
        set(FILE_PREFIX "${ARG_TEST_NAME}")
    else()
        set(FILE_PREFIX "${ARG_TEST_NAME}_task")
    endif()

    set(TASK_CRT0_OBJ "${SPATZ_OUTPUT_DIR}/${FILE_PREFIX}_crt0.o")
    set(TASK_ELF "${SPATZ_OUTPUT_DIR}/${FILE_PREFIX}.elf")
    set(TASK_BIN "${SPATZ_OUTPUT_DIR}/${FILE_PREFIX}.bin")
    set(TASK_DUMP "${SPATZ_OUTPUT_DIR}/${FILE_PREFIX}.dump")
    set(TASK_HEADER "${SPATZ_OUTPUT_DIR}/${FILE_PREFIX}_bin.h")

    set(TASK_INCLUDE_FLAGS "")
    foreach(INCLUDE_DIR ${MAGIA_TARGET_INCLUDE_DIRS})
        list(APPEND TASK_INCLUDE_FLAGS "-I${INCLUDE_DIR}")
    endforeach()
    foreach(INCLUDE_DIR ${ARG_INCLUDE_DIRS})
        list(APPEND TASK_INCLUDE_FLAGS "-I${INCLUDE_DIR}")
    endforeach()

    # --------------------------------------------------------------------------
    # Custom commands
    # --------------------------------------------------------------------------

    # Compile crt0.S [MAGIA/spatz/sw/Makefile: $(CRT0_OBJ)]
    add_custom_command(
        OUTPUT ${TASK_CRT0_OBJ}
        COMMAND ${SPATZ_CLANG}
            ${SPATZ_ARCH_FLAGS}
            -c -o ${TASK_CRT0_OBJ}
            ${SPATZ_CRT0_SRC}
        DEPENDS ${SPATZ_CRT0_SRC}
        COMMENT "[SPATZ] Compiling crt0..."
        VERBATIM
    )

    # Build ELF [MAGIA/spatz/sw/Makefile: $(ELF)]
    add_custom_command(
        OUTPUT ${TASK_ELF}
        COMMAND ${SPATZ_CLANG}
            ${SPATZ_COMPILE_FLAGS}
            ${SPATZ_CFLAGS_DEFINES}
            ${SPATZ_TASK_DEFINE}
            ${TASK_INCLUDE_FLAGS}
            ${SPATZ_LINK_FLAGS}
            -T${SPATZ_LINK_SCRIPT}
            -Wl,--defsym,__first_task=${ARG_FIRST_TASK_NAME}
            -o ${TASK_ELF}
            ${TASK_CRT0_OBJ}
            ${MAGIA_IO_SRC}
            ${ARG_TASK_SOURCES}
        DEPENDS
            ${TASK_CRT0_OBJ}
            ${MAGIA_IO_SRC}
            ${ARG_TASK_SOURCES}
            ${SPATZ_LINK_SCRIPT}
        COMMENT "[SPATZ] Building ELF..."
        VERBATIM
    )

    # Generate binary [MAGIA/spatz/sw/Makefile: $(BIN)]
    add_custom_command(
        OUTPUT ${TASK_BIN}
        COMMAND ${SPATZ_OBJCOPY} -O binary ${TASK_ELF} ${TASK_BIN}
        DEPENDS ${TASK_ELF}
        COMMENT "[SPATZ] Generating binary..."
        VERBATIM
    )

    # Generate disassembly [MAGIA/spatz/sw/Makefile: $(DUMP)]
    add_custom_command(
        OUTPUT ${TASK_DUMP}
        COMMAND ${SPATZ_OBJDUMP} -D -S ${TASK_ELF} > ${TASK_DUMP}
        DEPENDS ${TASK_ELF}
        COMMENT "[SPATZ] Generating disassembly..."
        VERBATIM
    )

    # Generate header [MAGIA/spatz/sw/Makefile: $(HEADER)]
    add_custom_command(
        OUTPUT ${TASK_HEADER}
        COMMAND python3 ${SPATZ_TASK_BIN2HEADER_SCRIPT}
            ${TASK_BIN} ${TASK_HEADER}
            --name ${ARG_TEST_NAME}_task_bin
            --section ${SPATZ_TASK_HEADER_SECTION}
            --address "${SPATZ_TASK_HEADER_ADDRESS}"
        COMMAND bash ${SPATZ_TASK_EXTRACT_SYMBOLS_SCRIPT}
            ${ARG_TEST_NAME} ${TASK_ELF} ${TASK_HEADER} ${SPATZ_OBJDUMP}
        DEPENDS
            ${TASK_BIN}
            ${TASK_ELF}
            ${SPATZ_TASK_BIN2HEADER_SCRIPT}
            ${SPATZ_TASK_EXTRACT_SYMBOLS_SCRIPT}
        COMMENT "[SPATZ] Generating header and extracting task symbols..."
        VERBATIM
    )

    # --------------------------------------------------------------------------
    # Target and exported variables
    # --------------------------------------------------------------------------
    add_custom_target(${ARG_TEST_NAME}_spatz_header ALL
        DEPENDS
            ${TASK_HEADER}
            ${TASK_BIN}
            ${TASK_DUMP}
    )

    set(${ARG_TEST_NAME}_SPATZ_HEADER ${TASK_HEADER} PARENT_SCOPE)
    set(${ARG_TEST_NAME}_TASK_BIN ${TASK_BIN} PARENT_SCOPE)
    set(${ARG_TEST_NAME}_TASK_HEADER ${TASK_HEADER} PARENT_SCOPE)
    set(${ARG_TEST_NAME}_SPATZ_OUTPUT_DIR ${SPATZ_OUTPUT_DIR} PARENT_SCOPE)
endfunction()

# add_cv32_executable_with_spatz(
#   TARGET_NAME <name>
#   SOURCES <src...>
#   [INCLUDE_DIRS <dirs...>]
# )
# Builds a CV32 executable linked with runtime/hal.
# Requires `add_spatz_task(TEST_NAME <same_name>)` to be called first in the
# same CMake scope.
function(add_cv32_executable_with_spatz)
    set(options)
    set(oneValueArgs TARGET_NAME)
    set(multiValueArgs SOURCES INCLUDE_DIRS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_TARGET_NAME)
        message(FATAL_ERROR "add_cv32_executable_with_spatz requires TARGET_NAME")
    endif()
    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "add_cv32_executable_with_spatz(${ARG_TARGET_NAME}) requires SOURCES")
    endif()
    if(NOT DEFINED ${ARG_TARGET_NAME}_SPATZ_HEADER)
        message(FATAL_ERROR "add_cv32_executable_with_spatz(${ARG_TARGET_NAME}) requires add_spatz_task(TEST_NAME ${ARG_TARGET_NAME}) first.")
    endif()

    # --------------------------------------------------------------------------
    # Target definition
    # --------------------------------------------------------------------------
    add_executable(${ARG_TARGET_NAME}
        ${CV32_CRT0_SRC}
        ${ARG_SOURCES}
        ${MAGIA_IO_SRC}
    )

    get_filename_component(SPATZ_HEADER_DIR "${${ARG_TARGET_NAME}_SPATZ_HEADER}" DIRECTORY)
    target_include_directories(${ARG_TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        ${MAGIA_TARGET_INCLUDE_DIRS}
        ${MAGIA_CV32_EXTRA_INCLUDE_DIRS}
        ${ARG_INCLUDE_DIRS}
        ${SPATZ_HEADER_DIR}
    )

    target_compile_options(${ARG_TARGET_NAME} PRIVATE ${CV32_COMPILE_FLAGS})
    target_link_options(${ARG_TARGET_NAME} PRIVATE ${CV32_LINK_FLAGS})
    target_link_libraries(${ARG_TARGET_NAME} PUBLIC runtime hal)

    # --------------------------------------------------------------------------
    # Post-build dump
    # --------------------------------------------------------------------------
    set(ELF_DUMP "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${ARG_TARGET_NAME}.s")
    add_custom_command(TARGET ${ARG_TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_OBJDUMP} -D -S $<TARGET_FILE:${ARG_TARGET_NAME}> > ${ELF_DUMP}
        VERBATIM
    )
endfunction()
