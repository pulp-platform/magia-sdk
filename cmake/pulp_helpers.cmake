# Helpers for building PULP cluster tasks and CV32 executables that embed them.

# add_pulp_task(
#   TEST_NAME <name>
#   TASK_SOURCES <src...>
#   [INCLUDE_DIRS <dirs...>]
# )
# Builds a PULP task ELF/BIN/DUMP, generates <name>_task_bin.h and exports
# per-test variables (<TEST_NAME>_PULP_HEADER, <TEST_NAME>_PULP_TASK_BIN, ...).
# All PULP harts boot at _start (pulp_crt0.S) and jump to main().
function(add_pulp_task)
    set(options)
    set(oneValueArgs TEST_NAME)
    set(multiValueArgs TASK_SOURCES INCLUDE_DIRS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_TEST_NAME)
        message(FATAL_ERROR "add_pulp_task requires TEST_NAME")
    endif()
    if(NOT ARG_TASK_SOURCES)
        message(FATAL_ERROR "add_pulp_task(${ARG_TEST_NAME}) requires TASK_SOURCES")
    endif()

    # --------------------------------------------------------------------------
    # Paths and names
    # --------------------------------------------------------------------------
    set(PULP_OUTPUT_DIR "${PULP_TASK_OUTPUT_ROOT}/${ARG_TEST_NAME}/pulp_task")
    file(MAKE_DIRECTORY "${PULP_OUTPUT_DIR}")

    string(REGEX MATCH "_task$" HAS_TASK_SUFFIX "${ARG_TEST_NAME}")
    if(HAS_TASK_SUFFIX)
        set(FILE_PREFIX "${ARG_TEST_NAME}")
    else()
        set(FILE_PREFIX "${ARG_TEST_NAME}_task")
    endif()

    set(TASK_CRT0_OBJ "${PULP_OUTPUT_DIR}/${FILE_PREFIX}_crt0.o")
    set(TASK_ELF      "${PULP_OUTPUT_DIR}/${FILE_PREFIX}.elf")
    set(TASK_BIN      "${PULP_OUTPUT_DIR}/${FILE_PREFIX}.bin")
    set(TASK_DUMP     "${PULP_OUTPUT_DIR}/${FILE_PREFIX}.dump")
    set(TASK_HEADER   "${PULP_OUTPUT_DIR}/${FILE_PREFIX}_bin.h")

    set(TASK_INCLUDE_FLAGS "")
    foreach(INCLUDE_DIR ${MAGIA_TARGET_INCLUDE_DIRS})
        list(APPEND TASK_INCLUDE_FLAGS "-I${INCLUDE_DIR}")
    endforeach()
    foreach(INCLUDE_DIR ${ARG_INCLUDE_DIRS})
        list(APPEND TASK_INCLUDE_FLAGS "-I${INCLUDE_DIR}")
    endforeach()

    message(NOTICE "FLAGS : ${TASK_INCLUDE_FLAGS}")

    set(TASK_UNDEF_FLAGS "")
    foreach(TASK_SOURCE ${ARG_TASK_SOURCES})
        get_filename_component(TASK_NAME "${TASK_SOURCE}" NAME_WE)
        list(APPEND TASK_UNDEF_FLAGS "-Wl,--undefined=${TASK_NAME}")
    endforeach()

    message(NOTICE "FLAGS : ${TASK_UNDEF_FLAGS}")

    # --------------------------------------------------------------------------
    # Custom commands
    # --------------------------------------------------------------------------

    # Compile pulp_crt0.S
    add_custom_command(
        OUTPUT ${TASK_CRT0_OBJ}
        COMMAND ${CMAKE_C_COMPILER}
            ${PULP_ARCH_FLAGS}
            -DPULP_CORE_COUNT=${PULP_CORE_COUNT}
            -c -o ${TASK_CRT0_OBJ}
            ${PULP_CRT0_SRC}
        DEPENDS ${PULP_CRT0_SRC}
        COMMENT "[PULP] Compiling crt0..."
        VERBATIM
    )

    # Build ELF (compile + link in one step, position-independent)
    add_custom_command(
        OUTPUT ${TASK_ELF}
        COMMAND ${CMAKE_C_COMPILER}
            ${PULP_COMPILE_FLAGS}
            ${PULP_CFLAGS_DEFINES}
            ${TASK_INCLUDE_FLAGS}
            ${PULP_LINK_FLAGS}
            ${TASK_UNDEF_FLAGS}
            -T${PULP_LINK_SCRIPT}
            -o ${TASK_ELF}
            ${TASK_CRT0_OBJ}
            ${MAGIA_IO_SRC}
            ${ARG_TASK_SOURCES}
        DEPENDS
            ${TASK_CRT0_OBJ}
            ${MAGIA_IO_SRC}
            ${ARG_TASK_SOURCES}
            ${PULP_LINK_SCRIPT}
        COMMENT "[PULP] Building ELF..."
        VERBATIM
    )

    # Generate flat binary
    add_custom_command(
        OUTPUT ${TASK_BIN}
        COMMAND ${CMAKE_OBJCOPY} -O binary ${TASK_ELF} ${TASK_BIN}
        DEPENDS ${TASK_ELF}
        COMMENT "[PULP] Generating binary..."
        VERBATIM
    )

    # Generate disassembly
    add_custom_command(
        OUTPUT ${TASK_DUMP}
        COMMAND ${CMAKE_OBJDUMP} -D -S ${TASK_ELF} > ${TASK_DUMP}
        DEPENDS ${TASK_ELF}
        COMMENT "[PULP] Generating disassembly..."
        VERBATIM
    )

    # Generate header and append _pulp_binary_start / PULP_BINARY_START
    add_custom_command(
        OUTPUT ${TASK_HEADER}
        COMMAND python3 ${PULP_TASK_BIN2HEADER_SCRIPT}
            ${TASK_BIN} ${TASK_HEADER}
            --name ${ARG_TEST_NAME}_task_bin
            --section ${PULP_TASK_HEADER_SECTION}
            --address "${PULP_TASK_HEADER_ADDRESS}"
        COMMAND bash ${PULP_TASK_EXTRACT_SYMBOLS_SCRIPT}
            ${ARG_TEST_NAME} ${TASK_ELF} ${TASK_HEADER} ${CMAKE_OBJDUMP}
        DEPENDS
            ${TASK_BIN}
            ${TASK_ELF}
            ${PULP_TASK_BIN2HEADER_SCRIPT}
            ${PULP_TASK_EXTRACT_SYMBOLS_SCRIPT}
        COMMENT "[PULP] Generating header and extracting symbols..."
        VERBATIM
    )

    # --------------------------------------------------------------------------
    # Target and exported variables
    # --------------------------------------------------------------------------
    string(MD5 _pulp_scope_hash "${CMAKE_CURRENT_SOURCE_DIR}_${ARG_TEST_NAME}")
    string(SUBSTRING "${_pulp_scope_hash}" 0 8 _pulp_scope_hash)
    set(_pulp_header_target "pulp_header_${ARG_TEST_NAME}_${_pulp_scope_hash}")

    add_custom_target(${_pulp_header_target} ALL
        DEPENDS
            ${TASK_HEADER}
            ${TASK_BIN}
            ${TASK_DUMP}
    )

    set(${ARG_TEST_NAME}_PULP_HEADER      ${TASK_HEADER}         PARENT_SCOPE)
    set(${ARG_TEST_NAME}_PULP_TARGET      ${_pulp_header_target} PARENT_SCOPE)
    set(${ARG_TEST_NAME}_PULP_TASK_BIN    ${TASK_BIN}            PARENT_SCOPE)
    set(${ARG_TEST_NAME}_PULP_OUTPUT_DIR  ${PULP_OUTPUT_DIR}     PARENT_SCOPE)
endfunction()

# add_cv32_executable_with_pulp(
#   TARGET_NAME <name>
#   SOURCES <src...>
#   [INCLUDE_DIRS <dirs...>]
# )
# Builds a CV32 executable that embeds a PULP task binary.
# Requires `add_pulp_task(TEST_NAME <same_name>)` to be called first in the
# same CMake scope.
function(add_cv32_executable_with_pulp)
    set(options)
    set(oneValueArgs TARGET_NAME)
    set(multiValueArgs SOURCES INCLUDE_DIRS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_TARGET_NAME)
        message(FATAL_ERROR "add_cv32_executable_with_pulp requires TARGET_NAME")
    endif()
    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "add_cv32_executable_with_pulp(${ARG_TARGET_NAME}) requires SOURCES")
    endif()
    if(NOT DEFINED ${ARG_TARGET_NAME}_PULP_HEADER)
        message(FATAL_ERROR "add_cv32_executable_with_pulp(${ARG_TARGET_NAME}) requires add_pulp_task(TEST_NAME ${ARG_TARGET_NAME}) first.")
    endif()

    # --------------------------------------------------------------------------
    # Target definition
    # --------------------------------------------------------------------------
    add_executable(${ARG_TARGET_NAME}
        ${CV32_CRT0_SRC}
        ${ARG_SOURCES}
        ${MAGIA_IO_SRC}
    )

    get_filename_component(PULP_HEADER_DIR "${${ARG_TARGET_NAME}_PULP_HEADER}" DIRECTORY)
    target_include_directories(${ARG_TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        ${MAGIA_TARGET_INCLUDE_DIRS}
        ${MAGIA_CV32_EXTRA_INCLUDE_DIRS}
        ${ARG_INCLUDE_DIRS}
        ${PULP_HEADER_DIR}
    )

    target_compile_options(${ARG_TARGET_NAME} PRIVATE ${PULP_COMPILE_FLAGS})
    target_link_options(${ARG_TARGET_NAME} PRIVATE ${PULP_LINK_FLAGS})
    target_link_libraries(${ARG_TARGET_NAME} PUBLIC runtime hal)

    add_dependencies(${ARG_TARGET_NAME} ${${ARG_TARGET_NAME}_PULP_TARGET})

    # --------------------------------------------------------------------------
    # Post-build dump
    # --------------------------------------------------------------------------
    set(ELF_DUMP "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${ARG_TARGET_NAME}.s")
    add_custom_command(TARGET ${ARG_TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_OBJDUMP} -D -S $<TARGET_FILE:${ARG_TARGET_NAME}> > ${ELF_DUMP}
        VERBATIM
    )
endfunction()

# add_cv32_executable_with_pulp_and_spatz(
#   TARGET_NAME <name>
#   SOURCES <src...>
#   [SPATZ_TEST_NAME <name>]   # defaults to TARGET_NAME
#   [PULP_TEST_NAME  <name>]   # defaults to TARGET_NAME
#   [INCLUDE_DIRS <dirs...>]
# )
# Builds a CV32 executable embedding both a Spatz task binary and a PULP task
# binary.  Requires add_spatz_task() and add_pulp_task() to be called first.
# Use SPATZ_TEST_NAME / PULP_TEST_NAME when the task TEST_NAMEs differ from
# the overall TARGET_NAME (e.g. hello_spatz / hello_pulp inside hello_spatz_pulp).
function(add_cv32_executable_with_pulp_and_spatz)
    set(options)
    set(oneValueArgs TARGET_NAME SPATZ_TEST_NAME PULP_TEST_NAME)
    set(multiValueArgs SOURCES INCLUDE_DIRS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_TARGET_NAME)
        message(FATAL_ERROR "add_cv32_executable_with_pulp_and_spatz requires TARGET_NAME")
    endif()
    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "add_cv32_executable_with_pulp_and_spatz(${ARG_TARGET_NAME}) requires SOURCES")
    endif()

    if(NOT ARG_SPATZ_TEST_NAME)
        set(ARG_SPATZ_TEST_NAME ${ARG_TARGET_NAME})
    endif()
    if(NOT ARG_PULP_TEST_NAME)
        set(ARG_PULP_TEST_NAME ${ARG_TARGET_NAME})
    endif()

    if(NOT DEFINED ${ARG_SPATZ_TEST_NAME}_SPATZ_HEADER)
        message(FATAL_ERROR "add_cv32_executable_with_pulp_and_spatz(${ARG_TARGET_NAME}) requires add_spatz_task(TEST_NAME ${ARG_SPATZ_TEST_NAME}) first.")
    endif()
    if(NOT DEFINED ${ARG_PULP_TEST_NAME}_PULP_HEADER)
        message(FATAL_ERROR "add_cv32_executable_with_pulp_and_spatz(${ARG_TARGET_NAME}) requires add_pulp_task(TEST_NAME ${ARG_PULP_TEST_NAME}) first.")
    endif()

    # --------------------------------------------------------------------------
    # Target definition
    # --------------------------------------------------------------------------
    add_executable(${ARG_TARGET_NAME}
        ${CV32_CRT0_SRC}
        ${ARG_SOURCES}
        ${MAGIA_IO_SRC}
    )

    get_filename_component(SPATZ_HEADER_DIR "${${ARG_SPATZ_TEST_NAME}_SPATZ_HEADER}" DIRECTORY)
    get_filename_component(PULP_HEADER_DIR  "${${ARG_PULP_TEST_NAME}_PULP_HEADER}"  DIRECTORY)
    target_include_directories(${ARG_TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        ${MAGIA_TARGET_INCLUDE_DIRS}
        ${MAGIA_CV32_EXTRA_INCLUDE_DIRS}
        ${ARG_INCLUDE_DIRS}
        ${SPATZ_HEADER_DIR}
        ${PULP_HEADER_DIR}
    )

    target_compile_options(${ARG_TARGET_NAME} PRIVATE ${PULP_COMPILE_FLAGS})
    target_link_options(${ARG_TARGET_NAME} PRIVATE ${PULP_LINK_FLAGS})
    target_link_libraries(${ARG_TARGET_NAME} PUBLIC runtime hal)

    add_dependencies(${ARG_TARGET_NAME}
        ${${ARG_SPATZ_TEST_NAME}_SPATZ_TARGET}
        ${${ARG_PULP_TEST_NAME}_PULP_TARGET}
    )

    # --------------------------------------------------------------------------
    # Post-build dump
    # --------------------------------------------------------------------------
    set(ELF_DUMP "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${ARG_TARGET_NAME}.s")
    add_custom_command(TARGET ${ARG_TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_OBJDUMP} -D -S $<TARGET_FILE:${ARG_TARGET_NAME}> > ${ELF_DUMP}
        VERBATIM
    )
endfunction()
