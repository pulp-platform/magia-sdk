# ============================================================================
# Spatz Task Compilation and Embedding Helpers
# ============================================================================
#
# This module provides two main CMake functions for building Spatz tasks
# and embedding them in CV32 executables:
#
#   1. add_spatz_task()
#      - Compiles Spatz task binary
#      - Generates header with embedded binary + task symbols
#      - Outputs to: build/bin/spatz_on_magia/<TEST_NAME>/spatz_task/
#
#   2. add_cv32_executable_with_spatz()
#      - Builds CV32 executable with embedded Spatz binary
#      - Generates disassembly and simulation stimuli
#      - Outputs to: build/bin/<executable>
#                    build/bin/spatz_on_magia/<TARGET_NAME>/stim/
# ============================================================================

include(${CMAKE_CURRENT_LIST_DIR}/spatz_config.cmake)

# Function: add_spatz_task
# Compiles Spatz task binary, generates header with binary array + task symbols, and stimuli
# Parameters:
#   TEST_NAME (required): Task identifier used for output file naming
#   TASK_SOURCES (required): List of source files to compile
#   CRT0_SRC: Path to CRT0 assembly (default: SPATZ_CRT0_SRC from config)
#   LINKER_SCRIPT: Path to linker script (default: SPATZ_LINK_SCRIPT from config)
#   OUTPUT_DIR: Base output directory (default: CMAKE_BINARY_DIR)
#   OUTPUT_SPATZ_DIR: Spatz task output directory
#     - If provided: use it directly
#     - If not provided: use ${OUTPUT_DIR}/bin/spatz_on_magia/${TEST_NAME}/spatz_task
#   OUTPUT_VAR: CMake variable to store TASK_HEADER path (default: ${TEST_NAME}_SPATZ_HEADER)
#   FIRST_TASK_NAME: Entry point function name (default: ${TEST_NAME}_task)
#   INCLUDE_DIRS: Additional include directories
function(add_spatz_task)
    set(options)
    set(oneValueArgs TEST_NAME CRT0_SRC LINKER_SCRIPT OUTPUT_DIR OUTPUT_SPATZ_DIR OUTPUT_VAR FIRST_TASK_NAME)
    set(multiValueArgs TASK_SOURCES INCLUDE_DIRS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Apply defaults for optional parameters
    if(NOT DEFINED ARG_CRT0_SRC)
        set(ARG_CRT0_SRC ${SPATZ_CRT0_SRC})
    endif()
    if(NOT DEFINED ARG_LINKER_SCRIPT)
        set(ARG_LINKER_SCRIPT ${SPATZ_LINK_SCRIPT})
    endif()
    if(NOT DEFINED ARG_OUTPUT_DIR)
        set(ARG_OUTPUT_DIR ${CMAKE_BINARY_DIR})
    endif()
    if(NOT DEFINED ARG_OUTPUT_VAR)
        set(ARG_OUTPUT_VAR ${ARG_TEST_NAME}_SPATZ_HEADER)
    endif()
    if(NOT DEFINED ARG_FIRST_TASK_NAME)
        set(ARG_FIRST_TASK_NAME ${ARG_TEST_NAME}_task)
    endif()

    if(NOT IS_ABSOLUTE "${ARG_CRT0_SRC}")
        set(ARG_CRT0_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_CRT0_SRC}")
    endif()
    if(NOT IS_ABSOLUTE "${ARG_LINKER_SCRIPT}")
        set(ARG_LINKER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_LINKER_SCRIPT}")
    endif()

    # Compute OUTPUT_SPATZ_DIR default if not provided
    if(NOT DEFINED ARG_OUTPUT_SPATZ_DIR)
        set(ARG_OUTPUT_SPATZ_DIR "${ARG_OUTPUT_DIR}/bin/spatz_on_magia/${ARG_TEST_NAME}/spatz_task")
    endif()
    set(SPATZ_OUTPUT_DIR "${ARG_OUTPUT_SPATZ_DIR}")

    # Create output directory if it doesn't exist
    file(MAKE_DIRECTORY "${SPATZ_OUTPUT_DIR}")

    # Compute filename prefix: avoid double "_task" if TEST_NAME already ends with "_task"
    # Logic: if TEST_NAME ends with "_task", use as-is; otherwise append "_task"
    string(REGEX MATCH "_task$" HAS_TASK_SUFFIX "${ARG_TEST_NAME}")
    if(HAS_TASK_SUFFIX)
        # TEST_NAME already ends with "_task" (e.g., "double_task" -> use "double_task" directly)
        set(FILE_PREFIX "${ARG_TEST_NAME}")
    else()
        # TEST_NAME doesn't end with "_task" (e.g., "hello_spatz" -> append "_task")
        set(FILE_PREFIX "${ARG_TEST_NAME}_task")
    endif()

    set(TASK_CRT0_OBJ "${SPATZ_OUTPUT_DIR}/${FILE_PREFIX}_crt0.o")
    set(TASK_ELF "${SPATZ_OUTPUT_DIR}/${FILE_PREFIX}.elf")
    set(TASK_BIN "${SPATZ_OUTPUT_DIR}/${FILE_PREFIX}.bin")
    set(TASK_HEADER "${SPATZ_OUTPUT_DIR}/${FILE_PREFIX}_bin.h")
    set(TASK_DUMP "${SPATZ_OUTPUT_DIR}/${FILE_PREFIX}.dump")

    set(INCLUDE_FLAGS "")
    # Add targets/magia_v2 include paths
    list(APPEND INCLUDE_FLAGS "-I${CMAKE_SOURCE_DIR}/targets/magia_v2/include")
    list(APPEND INCLUDE_FLAGS "-I${CMAKE_SOURCE_DIR}/targets/magia_v2/include/utils")
    list(APPEND INCLUDE_FLAGS "-I${CMAKE_SOURCE_DIR}/targets/magia_v2/include/addr_map")
    list(APPEND INCLUDE_FLAGS "-I${CMAKE_SOURCE_DIR}/targets/magia_v2/include/regs")
    # Add user-provided include directories
    foreach(INCLUDE_DIR ${ARG_INCLUDE_DIRS})
        list(APPEND INCLUDE_FLAGS "-I${INCLUDE_DIR}")
    endforeach()

    # Compile crt0.S [MAGIA/spatz/sw/Makefile: $(CRT0_OBJ)]
    add_custom_command(
        OUTPUT ${TASK_CRT0_OBJ}
        COMMAND ${SPATZ_CLANG}
            ${SPATZ_ARCH_FLAGS}
            -c -o ${TASK_CRT0_OBJ}
            ${ARG_CRT0_SRC}
        DEPENDS ${ARG_CRT0_SRC}
        COMMENT "[SPATZ] Compiling crt0..."
        VERBATIM
    )

    # Build ELF with crt0 + task sources [MAGIA/spatz/sw/Makefile: $(ELF)]
    add_custom_command(
        OUTPUT ${TASK_ELF}
        COMMAND ${SPATZ_CLANG}
            ${SPATZ_COMPILE_FLAGS}
            ${SPATZ_CFLAGS_DEFINES}
            -DSPATZ_TARGET
            ${INCLUDE_FLAGS}
            ${SPATZ_LINK_FLAGS}
            -T${ARG_LINKER_SCRIPT}
            -Wl,--defsym,__first_task=${ARG_FIRST_TASK_NAME}
            -o ${TASK_ELF}
            ${TASK_CRT0_OBJ}
            ${ARG_TASK_SOURCES}
        DEPENDS
            ${TASK_CRT0_OBJ}
            ${ARG_TASK_SOURCES}
            ${ARG_LINKER_SCRIPT}
        COMMENT "[SPATZ] Building ELF..."
        VERBATIM
    )

    # Generate binary from ELF [MAGIA/spatz/sw/Makefile: $(BIN)]
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


    set(BIN2HEADER_SCRIPT "${CMAKE_SOURCE_DIR}/scripts/bin2header.py")

    # Generate header with binary array [MAGIA/spatz/sw/Makefile: $(HEADER)]
    add_custom_command(
    OUTPUT ${TASK_HEADER}
    # Python Script
    COMMAND python3 ${BIN2HEADER_SCRIPT}
            ${TASK_BIN} ${TASK_HEADER}
            --name ${ARG_TEST_NAME}_task_bin
            --section .spatz_binary
            --address "dynamic (_spatz_binary_start)"
    # Extrac symbols
    COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/extract_task_symbols.sh
            ${ARG_TEST_NAME} ${TASK_ELF} ${TASK_HEADER} ${SPATZ_OBJDUMP}
    DEPENDS ${TASK_BIN} ${TASK_ELF} ${BIN2HEADER_SCRIPT} ${CMAKE_SOURCE_DIR}/scripts/extract_task_symbols.sh
    COMMENT "[SPATZ] Generating header and extracting task symbols..."
    VERBATIM
)

    add_custom_target(${ARG_TEST_NAME}_spatz_header ALL
        DEPENDS
            ${TASK_HEADER}
            ${TASK_BIN}
            ${TASK_DUMP}
    )

    # Export to parent scope
    set(${ARG_OUTPUT_VAR} ${TASK_HEADER} PARENT_SCOPE)
    set(${ARG_TEST_NAME}_TASK_BIN ${TASK_BIN} PARENT_SCOPE)
    set(${ARG_TEST_NAME}_TASK_HEADER ${TASK_HEADER} PARENT_SCOPE)
    set(${ARG_TEST_NAME}_SPATZ_OUTPUT_DIR ${SPATZ_OUTPUT_DIR} PARENT_SCOPE)
    # Also set generic SPATZ_HEADER for convenience when TEST_NAME==TARGET_NAME
    set(SPATZ_HEADER ${TASK_HEADER} PARENT_SCOPE)
    set(SPATZ_TASK_NAME ${ARG_TEST_NAME} PARENT_SCOPE)
endfunction()

# Function: add_cv32_executable_with_spatz
# Builds CV32 executable with embedded Spatz task binary and generates disassembly/stimuli
# Parameters:
#   TARGET_NAME (required): Executable name
#   SPATZ_HEADER (required): Path to Spatz task header file
#   SOURCES (required): List of source files to compile
#   CRT0_SRC: Path to CV32 CRT0 assembly (default: CV32_CRT0_SRC from config)
#   LINK_SCRIPT: Path to CV32 linker script (default: CV32_LINK_SCRIPT from config)
#   INCLUDE_DIRS: Additional include directories
# Output locations:
#   Executable: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET_NAME}
#   Disassembly: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET_NAME}.s
#   Stimuli files: ${CMAKE_BINARY_DIR}/bin/spatz_on_magia/${TARGET_NAME}/stim/
function(add_cv32_executable_with_spatz)
    set(options)
    set(oneValueArgs TARGET_NAME SPATZ_HEADER LINK_SCRIPT CRT0_SRC)
    set(multiValueArgs SOURCES INCLUDE_DIRS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Apply defaults for optional parameters
    if(NOT DEFINED ARG_CRT0_SRC)
        set(ARG_CRT0_SRC ${CV32_CRT0_SRC})
    endif()
    if(NOT DEFINED ARG_LINK_SCRIPT)
        set(ARG_LINK_SCRIPT ${CV32_LINK_SCRIPT})
    endif()

    if(NOT IS_ABSOLUTE "${ARG_CRT0_SRC}")
        set(ARG_CRT0_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_CRT0_SRC}")
    endif()
    if(ARG_LINK_SCRIPT AND NOT IS_ABSOLUTE "${ARG_LINK_SCRIPT}")
        set(ARG_LINK_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_LINK_SCRIPT}")
    endif()

    # Add io.c source to provide string functions for printf/prf.h
    set(IO_SRC "${CMAKE_SOURCE_DIR}/targets/magia_v2/src/io.c")
    add_executable(${ARG_TARGET_NAME} ${ARG_CRT0_SRC} ${ARG_SOURCES} ${IO_SRC})

    target_include_directories(${ARG_TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        ${CMAKE_SOURCE_DIR}/targets/magia_v2/include
        ${CMAKE_SOURCE_DIR}/targets/magia_v2/include/utils
        ${CMAKE_SOURCE_DIR}/targets/magia_v2/include/addr_map
        ${CMAKE_SOURCE_DIR}/targets/magia_v2/include/regs
        ${CMAKE_SOURCE_DIR}/hal/include
        ${CMAKE_SOURCE_DIR}/drivers/eventunit32/include
        ${ARG_INCLUDE_DIRS}
    )

    if(ARG_SPATZ_HEADER)
        get_filename_component(SPATZ_HEADER_DIR "${ARG_SPATZ_HEADER}" DIRECTORY)
        target_include_directories(${ARG_TARGET_NAME} PRIVATE ${SPATZ_HEADER_DIR})
    endif()

    # Compiler flags [MAGIA/Makefile: CC_OPTS]
    target_compile_options(${ARG_TARGET_NAME} PRIVATE
        ${CV32_MARCH}
        ${CV32_MABI}
        -D__riscv__
        -O2
        -g
        -Wall
        -Wextra
        -Wno-unused-parameter
        -Wno-unused-variable
        -Wno-unused-function
        -Wundef
        -fdata-sections
        -ffunction-sections
        -MMD
        -MP
    )

    # Linker flags [MAGIA/Makefile: LD_OPTS]
    target_link_options(${ARG_TARGET_NAME} PRIVATE
        ${CV32_MARCH}
        ${CV32_MABI}
        -D__riscv__
        -MMD
        -MP
        -nostartfiles
        -nostdlib
        -Wl,--gc-sections
    )

    if(ARG_LINK_SCRIPT)
        target_link_options(${ARG_TARGET_NAME} PRIVATE -T${ARG_LINK_SCRIPT})
    endif()

    if(ARG_SPATZ_HEADER)
        # Use SPATZ_TASK_NAME if available (set by add_spatz_task), otherwise extract from header filename
        if(DEFINED SPATZ_TASK_NAME)
            set(EXTRACTED_TEST_NAME ${SPATZ_TASK_NAME})
        else()
            # Fallback: extract TEST_NAME from header filename (legacy support)
            get_filename_component(HEADER_FILENAME "${ARG_SPATZ_HEADER}" NAME)
            string(REGEX REPLACE "_task_bin.*$" "" EXTRACTED_TEST_NAME "${HEADER_FILENAME}")
        endif()
        add_dependencies(${ARG_TARGET_NAME} ${EXTRACTED_TEST_NAME}_spatz_header)
    endif()

    # Post-build: disassembly and stimuli [MAGIA/Makefile: $(STIM_INSTR) $(STIM_DATA)]
    set(STIM_DIR "${CMAKE_BINARY_DIR}/bin/spatz_on_magia/${ARG_TARGET_NAME}/stim/")
    set(ELF_DUMP "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${ARG_TARGET_NAME}.s")
    set(ELF_OBJDUMP "${STIM_DIR}${ARG_TARGET_NAME}.objdump")
    set(ELF_ITB "${STIM_DIR}${ARG_TARGET_NAME}.itb")
    set(ELF_S19 "${STIM_DIR}${ARG_TARGET_NAME}.s19")
    set(ELF_TXT "${STIM_DIR}${ARG_TARGET_NAME}.txt")
    set(STIM_INSTR "${STIM_DIR}${ARG_TARGET_NAME}_stim_instr.txt")
    set(STIM_DATA "${STIM_DIR}${ARG_TARGET_NAME}_stim_data.txt")
    file(MAKE_DIRECTORY "${STIM_DIR}")

    add_custom_command(TARGET ${ARG_TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_OBJDUMP} -D -S $<TARGET_FILE:${ARG_TARGET_NAME}> > ${ELF_DUMP}
        COMMAND ${CMAKE_OBJDUMP} -h -S $<TARGET_FILE:${ARG_TARGET_NAME}> > ${ELF_OBJDUMP}
        COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/objdump2itb.py ${ELF_OBJDUMP} > ${ELF_ITB}
        COMMENT "[CV32] Generating disassembly and ITB for ${ARG_TARGET_NAME}"
        VERBATIM
    )

    add_custom_command(TARGET ${ARG_TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_OBJCOPY} --srec-len 1 --output-target=srec $<TARGET_FILE:${ARG_TARGET_NAME}> ${ELF_S19}
        COMMAND perl ${CMAKE_SOURCE_DIR}/scripts/parse_s19.pl ${ELF_S19} > ${ELF_TXT}
        COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/s19tomem.py ${ELF_TXT} ${STIM_INSTR} ${STIM_DATA}
        COMMENT "[CV32] Generating stimuli for ${ARG_TARGET_NAME}"
        VERBATIM
    )
endfunction()
