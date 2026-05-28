# Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)

# Core library
add_library(model_compute_core STATIC
  src/estimation_engine.cpp
  src/hardware_matcher.cpp
  src/calibration.cpp
)
target_include_directories(model_compute_core PUBLIC include)

# Python bindings
pybind11_add_module(model_compute bindings/pybind_module.cpp)
target_link_libraries(model_compute PRIVATE model_compute_core)

# Workaround: MinGW's GNU ld (binutils) crashes with exit code 116 when
# processing pybind11 object files containing many COMDAT sections via
# --whole-archive.  Use LLVM's lld linker instead, which handles this correctly.
if(MINGW)
  target_link_options(model_compute PRIVATE "-fuse-ld=lld")
endif()

# Tests
add_executable(test_engine tests/test_engine.cpp)
target_link_libraries(test_engine PRIVATE model_compute_core GTest::gtest_main)
if(MINGW OR UNIX)
  target_link_libraries(test_engine PRIVATE pthread)
endif()
if(MINGW)
  target_link_options(test_engine PRIVATE -static-libgcc -static-libstdc++)
endif()
include(GoogleTest)
gtest_discover_tests(test_engine)
