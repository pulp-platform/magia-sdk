# Magia-sdk
This repository contains the WIP development platform for the mesh-based architecture [MAGIA](https://github.com/pulp-platform/MAGIA/tree/main).
It provides useful tools for programming and running software applications on simulated MAGIA architectures for benchmarking, debugging and profiling.

Magia and Magia-SDK are developed as part of the [PULP project](https://pulp-platform.org/index.html), a joint effort between ETH Zurich and the University of Bologna.

## Getting started and Usage

The following *optional* parameters can be specified when running the make command:

`target_platform`: **magia_v1**|**magia_v2** (**Default**: magia_v2). Selects the target platform to build and run tests on. Magia V1 is the legacy mode using the CV32E40X core, whereas magia_v2 is the new platform using CV32E40P. No future support for magia_v1 is planned.

`build_mode`: **update**|**profile**|**synth** (**Default**: profile). Selects the mode that the MAGIA architecture is built.

`fsync_mode`: **stall**|**interrupt** (**Default** stall). Selects the Fractal Sync module synchronization behaviour.

`compiler`: **GCC_MULTILIB**|**GCC_PULP**|**LLVM** (**Default**: GCC_PULP). Selects the compiler to be used. LLVM is currently WIP. PULP is the risc-v 32 bits only toolchain NOT supporting floating point instructions, wheras the MULTILIB toolchain is the nightly risc-v one.

`platform`: **rtl**|**gvsoc**. Selects the simulation platform. GVSoC is currently WIP, some tests may fail.

`tiles`: **2**|**4**|**8**|**16** (**Default**: 2). Selects number of rows and columns for the mesh architecture.

`test_name`: Name of the test binary to be run.

`eval`: **0**|**1** (**Default**: 0). Activates printing of error values in the testsuite.

`gui`: **0**|**1** (**Default**: 0). Activates the graphic user interface on the simulator (rtl only).

`fast_sim`: **0**|**1** (**Default**: 0). Deactivates signal tracking for faster simulation (rtl only).

`redmule|fsync|idma_mm`: **0**|**1** (**Default**: 1). Uses memory mapped instructions for redmule|fsync|idma. **NOT SUPPORTED IN MAGIA_V1**

`stalling`: **0**|**1** (**Default**: 0). Activates stalling mode on tiles instead of using the event unit. **STALLING MUST BE SET ON 1 FOR MAGIA_V1**

`profile_cmp|cmi|cmo|snc`: **0**|**1** (**Default**: 0). Activates the profiling utilities for computing|comunication(input line)|comunication(output line)|synchronization

0. In case you are using this SDK as non-submodule: Clone the [MAGIA](https://github.com/pulp-platform/MAGIA/tree/main) repository:

    `git clone git@github.com:pulp-platform/MAGIA.git`

    Then modify the magia-sdk **Makefile** to point to the correct paths:

        MAGIA_DIR ?= path/to/MAGIA/repository

        BUILD_DIR ?= $(MAGIA_DIR)//work/sw/tests/$(test).c

    It is also required to have a cmake version >= 3.13.

1. Initialize the GVSoC submodule:

    `make gvsoc_init`
    
2. Build the Magia architecture (*this command may take time and return an error, please be patient.*):
        
    `make MAGIA <target_platform> <tiles> <build_mode> <fsync_mode>`

    And/Or the GVSoC module:

    `make gvsoc <tiles>`

    ***WARNING: PYTHON 3.12 IS MANDATORY***

    You can clean the MAGIA rtl and all the tests+logs by running:

    `make rtl-clean`

3. Make sure the RISC-V GCC compiler is installed and visible in the `$PATH` environment variable. You can check if and where the compiler is installed by running the following command on your root (`/`) directory:

    `find . ! -readable -prune -o -name "riscv64-unknown-elf-gcc" -print`

    Or (in case you want to use the 32-bit only PULP toolchain):

    `find . ! -readable -prune -o -name "riscv32-unknown-elf-gcc" -print`

    Then add the compiler to the `$PATH` environment variable with:

    `export PATH=<absolute path to directory containing the compiler binary>:$PATH`

    In case you don't have a toolchain, or the toolchain in your machine has compiler errors (such as requiring strange strange ISA extensions), you can build your own toolchain by following the steps listed [HERE](https://github.com/riscv-collab/riscv-gnu-toolchain). **Make sure you enable multilib to support 32-bit.** Despite having 64 in the name, the toolchain also supports 32-bit targets.

    In case you want to use a different toolchain, or want to specify a particular toolchain installed in your filesystem, you can edit the file *magia-sdk/cmake/toolchain_gcc_multilib.cmake* to point to your desired toolchain binary file. Make sure the ILP and API extensions in *CMakeLists.txt* are supported by the toolchain.

4. To compile and build the test binaries for a desired architecture run:

    `make clean build <target_platform> <tiles> <compiler> <eval>`

    To run one of the tests:

    `make run test=<test_name> <platform>`

***WARNING: YOU HAVE TO REBUILD BOTH RTL/GVSOC AND THE TEST BINARY EACH TIME YOU WANT TO TEST A MAGIA MESH WITH A DIFFERENT NUMBER OF TILES.*** 

If you want to run gvsoc or a binary from outside the magia-sdk directory you can edit the **GVSOC_ABS_PATH** and **BIN_ABS_PATH** option in Makefile or directly on the *run* command line.

To ensure a clean re-build of the RTL, you can run:

`make rtl-clean`

before building the RTL back using the `make MAGIA` command.

## Adding your own test

This SDK uses a nested CMakeList mechanism to build and check for dependencies.
To add your own test, you have to integrate a new test folder inside the **tests** directory.

1. Change directory to the desired architecture test folder

    `cd tests/<target_platform>/mesh`

2. Create a new test directory

    `mkdir <test_name>`

3. Modify the *CMakeList.txt* file in the current mesh directory, adding the following line:

    `add_subdirectory(<test_name>)`

    You can also *exclude* the generation of other tests binaries by commenting/deleting the lines for those tests.

4. Add to the *\<test_name\>* directory:

    1. A new CMakeList.txt file following this template:
    
            set(TEST_NAME <test_name>)

            file(GLOB_RECURSE TEST_SRCS
            "src/*.c"
            )

            add_executable(${TEST_NAME} ${TEST_SRCS})
            target_include_directories(${TEST_NAME} PUBLIC include)

            target_compile_options(${TEST_NAME}
            PRIVATE
            -O2
            )
            target_link_libraries(${TEST_NAME} PUBLIC runtime hal)

            add_custom_command(
                    TARGET ${TEST_NAME}
                    POST_BUILD
                    COMMAND ${CMAKE_OBJDUMP} -dhS $<TARGET_FILE:${TEST_NAME}> > $<TARGET_FILE:${TEST_NAME}>.s)
    
    2. An **src** directory containing your test's source (.c) files

    3. An **include** directory containing your test's header (.h) files

## Folder Structure

### README.md
This file.

### Makefile
Makefile script to run the sdk.

### CMakeLists.txt
Root CMakeLists file to compile and build the executable test/application binaries for one of the available architectures.

### targets
This directory contains the *startup routine*, *linker script*, *address map*, *register definitions*, *custom ISA instructions*, *MAGIA mesh and tile util instructions* for each available architecture.

### scripts
Contains scripts to automatize the test building and running.

### hal
Contains the weak definitions of this SDK APIs. These are the API instruction that should be used by the programmer when developing applications to be run on MAGIA. These instructions are then overloaded by the corresponding driver implementation specific for the chosen architecture. The APIs currently available are for controlling and using the *idma*, *redmule* and *fractalsync* modules.

### drivers
Contains the architecture-specific implementation and source code for the HAL APIs. Despite each implementation having different names, thanks to an aliasing system the programmer can use the same name for the same API instruction on different architectures.

### devices
Nothing there. 

If MAGIA ever evolves to have a host-offload mechanism, this folder will contain the trampoline functions.

### cmake
Contains utility files for *cmake* automatic compilation.

### gvsoc
A submodule containing the Germain Virtual System on Chip, built to simulate MAGIA (and other PULP-related platforms).

