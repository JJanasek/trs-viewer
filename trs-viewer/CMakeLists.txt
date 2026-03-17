cmake_minimum_required(VERSION 3.16)
project(trs-viewer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# Try Qt6 first, fall back to Qt5
find_package(Qt6 QUIET COMPONENTS Core Gui Widgets Svg)
if(Qt6_FOUND)
    set(QT_LIBS Qt6::Core Qt6::Gui Qt6::Widgets Qt6::Svg)
    message(STATUS "Using Qt6")
else()
    find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets Svg)
    set(QT_LIBS Qt5::Core Qt5::Gui Qt5::Widgets Qt5::Svg)
    message(STATUS "Using Qt5")
endif()

# ---------------------------------------------------------------------------
# OpenMP
# ---------------------------------------------------------------------------
find_package(OpenMP REQUIRED)
message(STATUS "OpenMP version: ${OpenMP_CXX_VERSION}")

# ---------------------------------------------------------------------------
# Eigen3 — try system package first, then FetchContent
# ---------------------------------------------------------------------------
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(Eigen3_FOUND)
    message(STATUS "Using system Eigen3 ${Eigen3_VERSION}")
else()
    message(STATUS "Eigen3 not found as system package — fetching 3.4.0 via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
    )
    set(EIGEN_BUILD_DOC   OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING     OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(eigen)
endif()

# ---------------------------------------------------------------------------
# Python3 + NumPy (for CPA leakage model scripting)
# ---------------------------------------------------------------------------
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
message(STATUS "Python3 version: ${Python3_VERSION}")
message(STATUS "NumPy include:   ${Python3_NumPy_INCLUDE_DIRS}")

set(SOURCES
    resources.qrc
    src/main.cpp
    src/mainwindow.cpp
    src/trs_file.cpp
    src/processing.cpp
    src/plot_widget.cpp
    src/ttest.cpp
    src/xcorr.cpp
    src/align.cpp
    src/heatmap_widget.cpp
    inc/mainwindow.h
    inc/plot_widget.h
    inc/heatmap_widget.h
    src/leakage_model.cpp
    src/cpa.cpp
    src/leakage_model_dialog.cpp
    inc/leakage_model.h
    inc/leakage_model_dialog.h
    inc/cpa.h
)

add_executable(trs-viewer ${SOURCES})

target_link_libraries(trs-viewer PRIVATE
    ${QT_LIBS}
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
    Python3::Python
    Python3::NumPy
)
target_include_directories(trs-viewer PRIVATE src inc ${Python3_NumPy_INCLUDE_DIRS})

target_compile_options(trs-viewer PRIVATE
    -O2
    -Wall
    -Wextra
    -Wno-unused-parameter
    -march=native
)
