# Spatz task compilation and embedding helpers

include(${CMAKE_CURRENT_LIST_DIR}/spatz_config.cmake)

# Function: add_spatz_task
# Compiles Spatz task binary, generates header with binary array + task symbols, and stimuli
function(add_spatz_task)
    set(options)
    set(oneValueArgs TEST_NAME CRT0_SRC LINKER_SCRIPT OUTPUT_DIR OUTPUT_VAR)
    set(multiValueArgs TASK_SOURCES FIRST_TASK_NAME)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT IS_ABSOLUTE "${ARG_CRT0_SRC}")
        set(ARG_CRT0_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_CRT0_SRC}")
    endif()
    if(NOT IS_ABSOLUTE "${ARG_LINKER_SCRIPT}")
        set(ARG_LINKER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_LINKER_SCRIPT}")
    endif()

    set(TASK_CRT0_OBJ "${ARG_OUTPUT_DIR}/${ARG_TEST_NAME}_crt0.o")
    set(TASK_ELF "${ARG_OUTPUT_DIR}/${ARG_TEST_NAME}_task.elf")
    set(TASK_BIN "${ARG_OUTPUT_DIR}/${ARG_TEST_NAME}_task.bin")
    set(TASK_HEADER "${ARG_OUTPUT_DIR}/${ARG_TEST_NAME}_task_bin.h")
    set(TASK_DUMP "${ARG_OUTPUT_DIR}/${ARG_TEST_NAME}_task.dump")

    # Compile crt0.S [MAGIA/spatz/sw/Makefile: $(CRT0_OBJ)]
    add_custom_command(
        OUTPUT ${TASK_CRT0_OBJ}
        COMMAND ${SPATZ_CLANG}
            ${SPATZ_COMPILE_FLAGS}
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
        COMMAND python3 ${BIN2HEADER_SCRIPT}
            ${TASK_BIN} ${TASK_HEADER}
            --name ${ARG_TEST_NAME}_task_bin
            --section .spatz_binary
            --address "dynamic (_spatz_binary_start)"
        DEPENDS ${TASK_BIN} ${BIN2HEADER_SCRIPT}
        COMMENT "[SPATZ] Generating header with binary array..."
        VERBATIM
    )

    # Extract task symbols and append to header [MAGIA/spatz/sw/Makefile: $(HEADER)]
    add_custom_command(
        OUTPUT ${TASK_HEADER}
        APPEND
        COMMAND bash -c "
            GUARD_NAME=\$(echo '${ARG_TEST_NAME}_TASK_BIN' | tr 'a-z' 'A-Z')
            sed -i \"/#endif.*__\$${GUARD_NAME}_H__/d\" ${TASK_HEADER}
            echo '' >> ${TASK_HEADER}
            echo '/* Binary start address - defined by CV32 linker */' >> ${TASK_HEADER}
            echo 'extern uint32_t _spatz_binary_start;' >> ${TASK_HEADER}
            echo '#define SPATZ_BINARY_START ((uint32_t)&_spatz_binary_start)' >> ${TASK_HEADER}
            echo '' >> ${TASK_HEADER}
            echo '/* Dispatcher loop address - for spatz_run() */' >> ${TASK_HEADER}
            DISP_ADDR=\$(${SPATZ_OBJDUMP} -t ${TASK_ELF} | grep 'dispatcher_loop\$\$' | awk '{print \$1}')
            echo \"#define SPATZ_DISPATCHER_LOOP (SPATZ_BINARY_START + 0x\$${DISP_ADDR})\" >> ${TASK_HEADER}
            echo '' >> ${TASK_HEADER}
            echo '/* Task function entry points - OFFSETS from SPATZ_BINARY_START */' >> ${TASK_HEADER}
            ${SPATZ_OBJDUMP} -t ${TASK_ELF} | grep '_task\$\$' | grep -v '__first_task\$\$' | awk '{print \$1, \$NF}' | while read addr name; do
                TASK_NAME=\$(echo \$name | tr 'a-z' 'A-Z')
                echo \"#define \$${TASK_NAME} (SPATZ_BINARY_START + 0x\$${addr})\" >> ${TASK_HEADER}
            done
            echo '' >> ${TASK_HEADER}
            echo \"#endif /* __\$${GUARD_NAME}_H__ */\" >> ${TASK_HEADER}
        "
        COMMENT "[SPATZ] Extracting task symbols..."
        VERBATIM
    )

    add_custom_target(${ARG_TEST_NAME}_spatz_header ALL
        DEPENDS
            ${TASK_HEADER}
            ${TASK_BIN}
            ${TASK_DUMP}
    )

    set(${ARG_OUTPUT_VAR} ${TASK_HEADER} PARENT_SCOPE)
    set(${ARG_TEST_NAME}_TASK_BIN ${TASK_BIN} PARENT_SCOPE)
endfunction()

# Function: add_cv32_executable_with_spatz
# Builds CV32 executable with embedded Spatz task binary
function(add_cv32_executable_with_spatz)
    set(options)
    set(oneValueArgs TARGET_NAME SPATZ_HEADER LINK_SCRIPT CRT0_SRC)
    set(multiValueArgs SOURCES INCLUDE_DIRS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT IS_ABSOLUTE "${ARG_CRT0_SRC}")
        set(ARG_CRT0_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_CRT0_SRC}")
    endif()
    if(ARG_LINK_SCRIPT AND NOT IS_ABSOLUTE "${ARG_LINK_SCRIPT}")
        set(ARG_LINK_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_LINK_SCRIPT}")
    endif()

    add_executable(${ARG_TARGET_NAME} ${ARG_CRT0_SRC} ${ARG_SOURCES})

    target_include_directories(${ARG_TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        ${CMAKE_SOURCE_DIR}/tests/utils
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
        -nostartfiles
        -nostdlib
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
        get_filename_component(HEADER_FILENAME "${ARG_SPATZ_HEADER}" NAME)
        string(REGEX MATCH "^([^_]+_[^_]+_[^_]+)" TEST_NAME "${HEADER_FILENAME}")
        add_dependencies(${ARG_TARGET_NAME} ${TEST_NAME}_spatz_header)
    endif()

    # Post-build: disassembly and stimuli [MAGIA/Makefile: $(STIM_INSTR) $(STIM_DATA)]
    set(ELF_DUMP "${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET_NAME}.dump")
    set(ELF_OBJDUMP "${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET_NAME}.objdump")
    set(ELF_ITB "${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET_NAME}.itb")
    set(ELF_S19 "${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET_NAME}.s19")
    set(ELF_TXT "${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET_NAME}.txt")
    set(STIM_INSTR "${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET_NAME}_stim_instr.txt")
    set(STIM_DATA "${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET_NAME}_stim_data.txt")

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
