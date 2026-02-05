# Spatz bootrom compilation helpers

include(${CMAKE_CURRENT_LIST_DIR}/spatz_config.cmake)

# Function: add_spatz_bootrom_binary
# Compiles Spatz bootrom
function(add_spatz_bootrom_binary)
    set(options)
    set(oneValueArgs BOOTROM_SRC LINKER_SCRIPT OUTPUT_DIR OUTPUT_VAR SV_OUTPUT)
    set(multiValueArgs)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_OUTPUT_DIR)
        set(ARG_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
    endif()

    if(NOT IS_ABSOLUTE "${ARG_BOOTROM_SRC}")
        set(ARG_BOOTROM_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_BOOTROM_SRC}")
    endif()
    if(NOT IS_ABSOLUTE "${ARG_LINKER_SCRIPT}")
        set(ARG_LINKER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_LINKER_SCRIPT}")
    endif()

    set(BOOTROM_ELF "${ARG_OUTPUT_DIR}/spatz_init.elf")
    set(BOOTROM_BIN "${ARG_OUTPUT_DIR}/spatz_init.bin")
    set(BOOTROM_DUMP "${ARG_OUTPUT_DIR}/spatz_init.dump")

    # Compile bootrom ELF [MAGIA/spatz/bootrom/Makefile: $(ELF)]
    add_custom_command(
        OUTPUT ${BOOTROM_ELF}
        COMMAND ${SPATZ_CLANG}
            ${SPATZ_COMPILE_FLAGS}
            ${SPATZ_CFLAGS_DEFINES}
            ${SPATZ_LINK_FLAGS}
            -T${ARG_LINKER_SCRIPT}
            -o ${BOOTROM_ELF}
            ${ARG_BOOTROM_SRC}
        DEPENDS ${ARG_BOOTROM_SRC} ${ARG_LINKER_SCRIPT}
        COMMENT "[SPATZ-BOOTROM] Compiling bootrom ELF..."
        VERBATIM
    )

    # Extract binary [MAGIA/spatz/bootrom/Makefile: $(BIN)]
    add_custom_command(
        OUTPUT ${BOOTROM_BIN}
        COMMAND ${SPATZ_OBJCOPY} -O binary ${BOOTROM_ELF} ${BOOTROM_BIN}
        DEPENDS ${BOOTROM_ELF}
        COMMENT "[SPATZ-BOOTROM] Generating bootrom binary..."
        VERBATIM
    )

    # Generate disassembly [MAGIA/spatz/bootrom/Makefile: $(DUMP)]
    add_custom_command(
        OUTPUT ${BOOTROM_DUMP}
        COMMAND ${SPATZ_OBJDUMP} -D ${BOOTROM_ELF} > ${BOOTROM_DUMP}
        DEPENDS ${BOOTROM_ELF}
        COMMENT "[SPATZ-BOOTROM] Generating disassembly..."
        VERBATIM
    )

    set(BOOTROM_TARGET_OUTPUTS ${BOOTROM_BIN} ${BOOTROM_DUMP})

    # Generate SystemVerilog bootrom module [MAGIA/spatz/bootrom/Makefile: "sv"]
    if(ARG_SV_OUTPUT)
        add_custom_command(
            OUTPUT ${ARG_SV_OUTPUT}
            COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/generate_spatz_bootrom.py
                ${BOOTROM_BIN}
                -o ${ARG_SV_OUTPUT}
            DEPENDS ${BOOTROM_BIN}
            COMMENT "[SPATZ-BOOTROM] Generating hardware bootrom: ${ARG_SV_OUTPUT}"
            VERBATIM
        )
        list(APPEND BOOTROM_TARGET_OUTPUTS ${ARG_SV_OUTPUT})
    endif()

    add_custom_target(spatz_bootrom ALL DEPENDS ${BOOTROM_TARGET_OUTPUTS})

    set(${ARG_OUTPUT_VAR} ${BOOTROM_BIN} PARENT_SCOPE)
endfunction()
