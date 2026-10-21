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
add_library(trs-core-test OBJECT
    ${CMAKE_SOURCE_DIR}/src/trs_file.cpp
    ${CMAKE_SOURCE_DIR}/src/processing.cpp
    ${CMAKE_SOURCE_DIR}/src/snr.cpp
    ${CMAKE_SOURCE_DIR}/src/ttest.cpp
    ${CMAKE_SOURCE_DIR}/src/ttest_run.cpp
    ${CMAKE_SOURCE_DIR}/src/ttest_bivariate.cpp
    ${CMAKE_SOURCE_DIR}/src/pattern_align.cpp
    ${CMAKE_SOURCE_DIR}/src/cpa.cpp
    ${CMAKE_SOURCE_DIR}/src/xcorr.cpp
    ${CMAKE_SOURCE_DIR}/src/align.cpp
    ${CMAKE_SOURCE_DIR}/src/spectral.cpp
    ${CMAKE_SOURCE_DIR}/src/chain.cpp
)

target_include_directories(trs-core-test PUBLIC
    ${CMAKE_SOURCE_DIR}/inc
    ${CMAKE_SOURCE_DIR}/src
)

# chain.cpp needs Qt Core (QString, QJson*) — but only Core, not the widget
# stack, which is exactly why chain.cpp keeps its step data model separate from
# MainWindow's dialogs and so can be tested headlessly.
target_link_libraries(trs-core-test PUBLIC
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
    ZLIB::ZLIB
    ${QT_LIBS_CORE}
)

target_compile_options(trs-core-test PRIVATE
    -O1 -Wall -Wextra -Wno-unused-parameter
)

if(ENABLE_SANITIZERS)
    message(STATUS "Tests: AddressSanitizer + LeakSanitizer enabled")
    target_compile_options(trs-core-test PUBLIC -fsanitize=address,leak -fno-omit-frame-pointer -g)
    target_link_options(   trs-core-test PUBLIC -fsanitize=address,leak)
endif()

# ---------------------------------------------------------------------------
# Helper: register a test executable and auto-discover its tests via ctest
# ---------------------------------------------------------------------------
function(add_trs_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE trs-core-test GTest::gtest_main)
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
add_trs_test(test_ttest_live)
add_trs_test(test_ttest_order2)
add_trs_test(test_ttest_bivariate)
add_trs_test(test_pattern_align)
add_trs_test(test_trs_file)
add_trs_test(test_cpa)
add_trs_test(test_align)
add_trs_test(test_xcorr)
add_trs_test(test_spectral)
add_trs_test(test_chain)

# Plain vs live t-test timing on a real TRS file — a manual benchmark, not a
# test, so it is built but not registered with ctest. Uses -O2 core sources
# rather than the -O1 trs-core-test objects so the numbers mean something.
add_executable(bench_ttest_live bench_ttest_live.cpp
    ${CMAKE_SOURCE_DIR}/src/trs_file.cpp
    ${CMAKE_SOURCE_DIR}/src/processing.cpp
    ${CMAKE_SOURCE_DIR}/src/spectral.cpp
    ${CMAKE_SOURCE_DIR}/src/ttest.cpp
    ${CMAKE_SOURCE_DIR}/src/ttest_run.cpp
    ${CMAKE_SOURCE_DIR}/src/align.cpp
    ${CMAKE_SOURCE_DIR}/src/xcorr.cpp)
target_include_directories(bench_ttest_live PRIVATE ${CMAKE_SOURCE_DIR}/inc ${CMAKE_SOURCE_DIR}/src)
target_compile_options(bench_ttest_live PRIVATE -O2 -march=native)
target_link_libraries(bench_ttest_live PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen ZLIB::ZLIB ${QT_LIBS_CORE})

# HeatmapWidget is a QWidget, so unlike the rest of the suite this target needs
# the Qt widget stack and a QApplication (created offscreen in the test's own
# main). Kept separate so the headless core tests stay free of it.
# The header must be listed too, or AUTOMOC never sees the Q_OBJECT and the
# signals link as undefined.
add_executable(test_heatmap test_heatmap.cpp
    ${CMAKE_SOURCE_DIR}/src/heatmap_widget.cpp
    ${CMAKE_SOURCE_DIR}/inc/heatmap_widget.h)
set_target_properties(test_heatmap PROPERTIES AUTOMOC ON)
target_link_libraries(test_heatmap PRIVATE GTest::gtest ${QT_LIBS})
target_include_directories(test_heatmap PRIVATE ${CMAKE_SOURCE_DIR}/inc)
if(ENABLE_SANITIZERS)
    target_compile_options(test_heatmap PRIVATE -fsanitize=address,leak -fno-omit-frame-pointer -g)
    target_link_options(   test_heatmap PRIVATE -fsanitize=address,leak)
endif()
gtest_discover_tests(test_heatmap)
