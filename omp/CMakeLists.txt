# Device-side OpenMP runtime for MAGIA (port of HERO libhero/src/device).
# OBJECT, not STATIC: main() is defined here and called by the crt0 of `runtime`.

option(OMP_SELF_TEST "omptarget: the device also acts as a fake host" ON)

# Provides HERO_FOUND and omp_hero_kernels()
include(${CMAKE_CURRENT_LIST_DIR}/hero_magia.cmake)

add_library(omptarget OBJECT
  src/omptarget_magia.c
  src/sw_mailbox.c
)

target_include_directories(omptarget PUBLIC include)

if(OMP_SELF_TEST)
  target_compile_definitions(omptarget PRIVATE OFFLOAD_SELF_TEST)
endif()

# Use the include dirs and compile options of `runtime` without linking it:
# runtime lists crt0.S and io.c as PUBLIC sources, so linking it here would
# compile them into omptarget and again into the test (duplicate symbols).
target_include_directories(omptarget PRIVATE $<TARGET_PROPERTY:runtime,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_options(omptarget PRIVATE $<TARGET_PROPERTY:runtime,INTERFACE_COMPILE_OPTIONS>)

# Workaround for an internal compiler error in the SDK CORE-V GCC 14.1: with
# xcvmem it emits register+register stores (cv.sb) in the byte copy of
# rb_device_put (ringbuf.h) and crashes. Files including ringbuf.h are built
# without xcvmem. The last -march on the command line wins.
string(REPLACE "_xcvmem" "" OMP_ISA "${ISA}")
set(OMP_ISA "${OMP_ISA}" CACHE INTERNAL "ISA for the OpenMP code (without xcvmem)")
set_source_files_properties(src/omptarget_magia.c src/sw_mailbox.c
  PROPERTIES COMPILE_OPTIONS "-march=${OMP_ISA}")
