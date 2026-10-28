# ---------------------------------------------------------------------------
# Test suite for trs-viewer (core units, the heatmap widget, and the GUI soak test)
# Run with:  cmake -DBUILD_TESTS=ON ..  then  ctest --output-on-failure
# Memory errors and leaks: tools/run_asan.sh (configures -DTRS_SANITIZE=address,undefined)
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

# Sanitizers: configure with -DTRS_SANITIZE=address,undefined (top-level
# CMakeLists.txt), which covers the app and every target here.

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
    ${CMAKE_SOURCE_DIR}/src/dcor.cpp
    ${CMAKE_SOURCE_DIR}/src/ttest_dcor.cpp
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

# ---------------------------------------------------------------------------
# Helper: register a test executable and auto-discover its tests via ctest
# ---------------------------------------------------------------------------
function(add_trs_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE trs-core-test GTest::gtest_main)
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/inc)
    gtest_discover_tests(${name})
endfunction()

add_trs_test(test_processing)
add_trs_test(test_snr)
add_trs_test(test_ttest)
add_trs_test(test_ttest_live)
add_trs_test(test_ttest_order2)
add_trs_test(test_ttest_bivariate)
add_trs_test(test_dcor)
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
gtest_discover_tests(test_heatmap)

# The whole app driven offscreen through its menus and dialogs, one scenario
# per test (tests/gui_soak.cpp): the only test that runs MainWindow, the plots
# and the jobs together. Each scenario loops, and fails if memory keeps
# growing after the first loop. tools/run_asan.sh, run_valgrind.sh and
# run_heaptrack.sh run it under the memory tools.
add_executable(gui_soak gui_soak.cpp)
target_link_libraries(gui_soak PRIVATE trs-viewer-objs)
set(GUI_SOAK_SCENARIOS ttest average fft cpa cpa_top_curves plot_whole_trace export_npz snapshots combined quit_dialogs
    crash_quit_xcorr crash_job_during_modal crash_close_tab_drawing crash_nonmodal_other_tab)
foreach(sc ${GUI_SOAK_SCENARIOS})
    add_test(NAME gui_soak.${sc} COMMAND gui_soak --scenario ${sc} --loops 3)
    set_tests_properties(gui_soak.${sc} PROPERTIES
        ENVIRONMENT "QT_QPA_PLATFORM=offscreen;TRS_NO_CRASH_HANDLER=1"
        TIMEOUT 600)
endforeach()
