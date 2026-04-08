# Spatz bootrom compilation helpers

# add_spatz_bootrom()
# Builds the Spatz bootrom artifacts (ELF/BIN/DUMP).
function(add_spatz_bootrom)
    file(MAKE_DIRECTORY "${SPATZ_BOOTROM_OUTPUT_DIR}")

    # Compile bootrom ELF [MAGIA/spatz/bootrom/Makefile: $(ELF)]
    add_custom_command(
        OUTPUT ${SPATZ_BOOTROM_ELF}
        COMMAND ${SPATZ_CLANG}
            ${SPATZ_COMPILE_FLAGS}
            ${SPATZ_CFLAGS_DEFINES}
            ${SPATZ_LINK_FLAGS}
            -T${SPATZ_BOOTROM_LINK_SCRIPT}
            -o ${SPATZ_BOOTROM_ELF}
            ${SPATZ_BOOTROM_SRC}
        DEPENDS ${SPATZ_BOOTROM_SRC} ${SPATZ_BOOTROM_LINK_SCRIPT}
        COMMENT "[SPATZ-BOOTROM] Compiling bootrom ELF..."
        VERBATIM
    )

    # Extract binary [MAGIA/spatz/bootrom/Makefile: $(BIN)]
    add_custom_command(
        OUTPUT ${SPATZ_BOOTROM_BIN}
        COMMAND ${SPATZ_OBJCOPY} -O binary ${SPATZ_BOOTROM_ELF} ${SPATZ_BOOTROM_BIN}
        DEPENDS ${SPATZ_BOOTROM_ELF}
        COMMENT "[SPATZ-BOOTROM] Generating bootrom binary..."
        VERBATIM
    )

    # Generate disassembly [MAGIA/spatz/bootrom/Makefile: $(DUMP)]
    add_custom_command(
        OUTPUT ${SPATZ_BOOTROM_DUMP}
        COMMAND ${SPATZ_OBJDUMP} -D ${SPATZ_BOOTROM_ELF} > ${SPATZ_BOOTROM_DUMP}
        DEPENDS ${SPATZ_BOOTROM_ELF}
        COMMENT "[SPATZ-BOOTROM] Generating disassembly..."
        VERBATIM
    )

    add_custom_target(spatz_bootrom ALL DEPENDS ${SPATZ_BOOTROM_BIN} ${SPATZ_BOOTROM_DUMP})
    set(SPATZ_BOOTROM_BIN ${SPATZ_BOOTROM_BIN} PARENT_SCOPE)
endfunction()
