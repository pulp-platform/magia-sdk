# Tests for the device-side OpenMP runtime (omp/).

# The kernels are generated from "#pragma omp target": requires the HERO toolchain
if(HERO_FOUND)
  add_subdirectory(offload)
else()
  message(STATUS "[MAGIA-SDK] HERO not found in ${HERO_INSTALL}: skipping test_omp_offload "
                 "(set HERO_INSTALL to your HERO2/install)")
endif()
