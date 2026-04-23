# ---------------------------------------------------------------------------
# Test suite for trs-viewer non-GUI components
# Run with:  cmake -DBUILD_TESTS=ON ..  then  ctest --output-on-failure
# Memory leaks: cmake -DBUILD_TESTS=ON -DENABLE_SANITIZERS=ON ..
# ---------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

# ---------------------------------------------------------------------------
# GoogleTest — fetched via FetchContent
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# ---------------------------------------------------------------------------
# Optional AddressSanitizer + LeakSanitizer
# Set ENABLE_SANITIZERS=ON for memory-error / leak detection.
# ---------------------------------------------------------------------------
option(ENABLE_SANITIZERS "Build tests with AddressSanitizer + LeakSanitizer" OFF)

# ---------------------------------------------------------------------------
# Non-GUI core object library shared by all test executables.
# Excludes: main.cpp, all Qt widgets, leakage_model (Python-only), resources.
# ---------------------------------------------------------------------------
add_library(trs-core OBJECT
    ${CMAKE_SOURCE_DIR}/src/trs_file.cpp
    ${CMAKE_SOURCE_DIR}/src/processing.cpp
    ${CMAKE_SOURCE_DIR}/src/snr.cpp
    ${CMAKE_SOURCE_DIR}/src/ttest.cpp
    ${CMAKE_SOURCE_DIR}/src/cpa.cpp
    ${CMAKE_SOURCE_DIR}/src/xcorr.cpp
    ${CMAKE_SOURCE_DIR}/src/align.cpp
)

target_include_directories(trs-core PUBLIC
    ${CMAKE_SOURCE_DIR}/inc
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(trs-core PUBLIC
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
    ZLIB::ZLIB
)

target_compile_options(trs-core PRIVATE
    -O1 -Wall -Wextra -Wno-unused-parameter
)

if(ENABLE_SANITIZERS)
    message(STATUS "Tests: AddressSanitizer + LeakSanitizer enabled")
    target_compile_options(trs-core PUBLIC -fsanitize=address,leak -fno-omit-frame-pointer -g)
    target_link_options(   trs-core PUBLIC -fsanitize=address,leak)
endif()

# ---------------------------------------------------------------------------
# Helper: register a test executable and auto-discover its tests via ctest
# ---------------------------------------------------------------------------
function(add_trs_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE trs-core GTest::gtest_main)
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/inc)
    if(ENABLE_SANITIZERS)
        target_compile_options(${name} PRIVATE -fsanitize=address,leak -fno-omit-frame-pointer -g)
        target_link_options(   ${name} PRIVATE -fsanitize=address,leak)
    endif()
    gtest_discover_tests(${name})
endfunction()

add_trs_test(test_processing)
add_trs_test(test_snr)
add_trs_test(test_ttest)
add_trs_test(test_trs_file)
add_trs_test(test_cpa)
add_trs_test(test_align)
add_trs_test(test_xcorr)
