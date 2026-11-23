# HERO toolchain used to compile the "#pragma omp target" regions into device
# code for MAGIA (CV32E40P control core, rv32imc). CMake port of the HERO flow:
#   - HERO2/apps/omp/common/devices.mk and default.mk (compilation chain)
#   - HeroDevice.cpp (-hero1-* options)
# LLVM is not modified: all options are passed on the command line.
# Variables are CACHE because omp_hero_kernels() is called from the test
# directories, which do not see the normal variables of omp/.
#
# Toolchain build: source HERO2/scripts/setenv.sh && make hero-tc-llvm -> HERO2/install

# HERO_INSTALL defaults to a local path of the author's machine: point it to
# your HERO2/install, e.g. `export HERO_INSTALL=<path>` before `make build`
# (or `cmake -DHERO_INSTALL=<path>`).
if(DEFINED ENV{HERO_INSTALL})
  set(_hero_default "$ENV{HERO_INSTALL}")
else()
  set(_hero_default "/data2/fseratini/HERO2/install")
endif()
set(HERO_INSTALL "${_hero_default}" CACHE PATH "HERO toolchain install dir (clang/llc/hc-omp-pass)")

set(HERO_CLANG "${HERO_INSTALL}/bin/clang" CACHE INTERNAL "")
set(HERO_COB "${HERO_INSTALL}/bin/clang-offload-bundler" CACHE INTERNAL "")
set(HERO_HOP "${HERO_INSTALL}/bin/hc-omp-pass" CACHE INTERNAL "")            # runs the HERCULES passes
set(HERO_LLC "${HERO_INSTALL}/bin/llc" CACHE INTERNAL "")

# TODO(RTEMS host): replace the HERO Linux host triple with the RTEMS one once
# the host side of kernels_omp.c is compiled and used (discarded today).
set(HERO_HOST_TRIPLE "riscv64-hero-linux-gnu" CACHE INTERNAL "")
set(HERO_DEV_TRIPLE "riscv32-hero-hero1-elf" CACHE INTERNAL "")

# The code is already rv32imc, but without -mattr=+m,+c llc marks the object as
# plain rv32i (no RVC flag) and objdump cannot decode it.
set(HERO_MAGIA_MARCH "rv32imc" CACHE INTERNAL "")
set(HERO_MAGIA_LLC_ATTR "+m,+c" CACHE INTERNAL "")

# Used for headers only (stdint.h), no library is linked from it.
# TODO: switch to an rv32imc-ilp32 sysroot.
set(HERO_MAGIA_SYSROOT "${HERO_INSTALL}/rv32imafd-ilp32d/riscv32-unknown-elf"
    CACHE PATH "Device sysroot for the HERO clang (headers only)")

# Read by tests/magia/omp to decide whether to build the test
if(EXISTS "${HERO_CLANG}" AND EXISTS "${HERO_HOP}")
  set(HERO_FOUND TRUE CACHE INTERNAL "HERO toolchain found")
else()
  set(HERO_FOUND FALSE CACHE INTERNAL "HERO toolchain found")
endif()

# omp_hero_kernels(<out_obj> <src.c> <include_dir>)
#
# Builds <out_obj> from <src.c> with the same chain as HERO2/apps/omp/common/default.mk:
#   1) clang -emit-llvm          -> <base>.ll             (host + device bundle)
#   2) clang-offload-bundler     -> <base>-hero1.ll       (device only)
#   3) OmpKernelWrapper          -> <base>-hero1.TMP.1.ll (f(uint64_t argbuf) wrappers)
#   4) OmpHostPointerLegalizer   -> <base>-hero1.TMP.2.ll (host accesses -> hero_load/store)
#   5) llc                       -> <out_obj>             (rv32imc)
# Intermediate files are kept in the test build directory for inspection.
function(omp_hero_kernels out_obj src inc_dir)
  get_filename_component(base "${src}" NAME_WE)
  set(b "${CMAKE_CURRENT_BINARY_DIR}/${base}")

  set(dev_flags
    --hero1-sysroot=${HERO_MAGIA_SYSROOT}
    -hero1-march=${HERO_MAGIA_MARCH}
    -hero1-D__HERO_DEV
    -hero1-I${inc_dir}
  )

  add_custom_command(
    OUTPUT  ${b}.ll
    COMMAND ${HERO_CLANG} -c -emit-llvm -S -O2 -target ${HERO_HOST_TRIPLE}
            -fopenmp=libomp -fopenmp-targets=${HERO_DEV_TRIPLE} ${dev_flags}
            ${src} -o ${b}.ll
    DEPENDS ${src} ${inc_dir}/hero_64_magia.h
    COMMENT "[HERO] clang -fopenmp ${base}.c"
    VERBATIM)

  add_custom_command(
    OUTPUT  ${b}-hero1.ll ${b}-host.ll
    COMMAND ${HERO_COB} -inputs=${b}.ll -outputs=${b}-host.ll,${b}-hero1.ll -type=ll
            -targets=host-${HERO_HOST_TRIPLE},openmp-${HERO_DEV_TRIPLE} -unbundle
    DEPENDS ${b}.ll
    COMMENT "[HERO] clang-offload-bundler -unbundle"
    VERBATIM)

  add_custom_command(
    OUTPUT  ${b}-hero1.TMP.1.ll
    COMMAND ${CMAKE_COMMAND} -E env HERO_INSTALL=${HERO_INSTALL}
            ${HERO_HOP} ${b}-hero1.ll OmpKernelWrapper "HERCULES-omp-kernel-wrapper" ${b}-hero1.TMP.1.ll
    DEPENDS ${b}-hero1.ll
    COMMENT "[HERO] OmpKernelWrapper"
    VERBATIM)

  add_custom_command(
    OUTPUT  ${b}-hero1.TMP.2.ll
    COMMAND ${CMAKE_COMMAND} -E env HERO_INSTALL=${HERO_INSTALL}
            ${HERO_HOP} ${b}-hero1.TMP.1.ll OmpHostPointerLegalizer "HERCULES-omp-host-pointer-legalizer" ${b}-hero1.TMP.2.ll
    DEPENDS ${b}-hero1.TMP.1.ll
    COMMENT "[HERO] OmpHostPointerLegalizer"
    VERBATIM)

  add_custom_command(
    OUTPUT  ${out_obj}
    COMMAND ${HERO_LLC} -filetype=obj -mattr=${HERO_MAGIA_LLC_ATTR}
            ${b}-hero1.TMP.2.ll -o ${out_obj}
    DEPENDS ${b}-hero1.TMP.2.ll
    COMMENT "[HERO] llc -> ${base}.o"
    VERBATIM)

  set_source_files_properties(${out_obj} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
endfunction()
