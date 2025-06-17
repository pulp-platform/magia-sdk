# Copyright 2024-2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Moritz Scherer <scheremo@iis.ee.ethz.ch>
# Alberto Dequino <alberto.dequino@unibo.it>

macro(add_magia_executable name)
    add_executable(${ARGV})
    add_custom_command(
        TARGET ${name}
        POST_BUILD
        COMMAND ${CMAKE_OBJDUMP} -dhS $<TARGET_FILE:${name}> > $<TARGET_FILE:${name}>.s)
endmacro()

macro(add_magia_test name)
  add_magia_executable(${ARGV})
  if(TEST_MODE STREQUAL "simulation")
    add_test(NAME ${name} COMMAND ${SIMULATION_BINARY} +BINARY=$<TARGET_FILE:${name}> +PRELMODE=${PRELOAD_MODE_INT})
  endif()
endmacro()

macro(add_target_source name)
    if(NOT ${name} IN_LIST AVAILABLE_TARGETS)
        message(FATAL_ERROR "Invalid value for TARGET_PLATFORM: Got ${TARGET_PLATFORM}")
    endif()

    if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/${name})
        add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/${name})
    else()
        message(WARNING "Path ${CMAKE_CURRENT_LIST_DIR}/${name} does not exist")
    endif()
endmacro()

function(add_magia_subdirectories target_platform category mappings)
    # Initialize included folders
    set(included_folders "")

    # Process mappings
    foreach(mapping IN LISTS mappings)
        string(FIND "${mapping}" ":" delim_pos)
        if(delim_pos EQUAL -1)
        message(WARNING "[MAGIA-SDK] Invalid mapping entry: '${mapping}'. Skipping.")
        continue()
        endif()

        # Extract key and value
        string(SUBSTRING "${mapping}" 0 ${delim_pos} key)
        math(EXPR value_start "${delim_pos} + 1")
        string(SUBSTRING "${mapping}" ${value_start} -1 value)

        if(key STREQUAL "${target_platform}")
        list(APPEND included_folders ${value})
        break()
        endif()
    endforeach()

    string(REPLACE "," ";" included_folders "${included_folders}")

    # Align output with padding
    string(LENGTH "[MAGIA-SDK] Enabled ${category}s" category_prefix_length)
    math(EXPR padding_length "36 - ${category_prefix_length}")
    if(padding_length GREATER 0)
        string(REPEAT " " ${padding_length} padding)
    else()
        set(padding "")
    endif()

    # Debug: Print the folders being included
    message(STATUS "[MAGIA-SDK] Enabled ${category}s${padding}: ${included_folders}")

    # Add subdirectories, checking for a valid CMakeLists.txt
    foreach(folder IN LISTS included_folders)
        if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/${folder}/CMakeLists.txt)
        add_subdirectory(${folder})
        else()
        message(WARNING "[MAGIA-SDK] ${category} folder '${folder}' does not contain a valid CMakeLists.txt. Skipping.")
        endif()
    endforeach()
endfunction()