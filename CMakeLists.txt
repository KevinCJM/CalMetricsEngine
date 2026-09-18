cmake_minimum_required(VERSION 3.20...4.2)

if(CMAKE_HOST_APPLE AND NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET AND "$ENV{MACOSX_DEPLOYMENT_TARGET}" STREQUAL "")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum supported macOS")
endif()

project(calmetrics_engine LANGUAGES CXX)

option(CALMETRICS_ENGINE_BUILD_PYTHON "Build the Python extension" ON)
option(CALMETRICS_ENGINE_BUILD_TESTS "Build standalone native tests" OFF)
option(CALMETRICS_ENGINE_SANITIZE "Enable address/undefined-behavior sanitizers" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

find_package(Threads REQUIRED)

add_library(calmetrics_engine_finance STATIC
    cpp/finance/statistics.cpp
    cpp/finance/drawdown.cpp
    cpp/finance/streaks.cpp
    cpp/finance/rolling.cpp
)
target_include_directories(calmetrics_engine_finance PUBLIC cpp/include)
target_link_libraries(calmetrics_engine_finance PUBLIC Threads::Threads)

set(calmetrics_engine_targets calmetrics_engine_finance)

if(CALMETRICS_ENGINE_BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_native cpp/bindings.cpp)
    target_link_libraries(_native PRIVATE calmetrics_engine_finance)

    if(NOT DEFINED SKBUILD_PROJECT_VERSION)
        message(FATAL_ERROR "Build Python via python -m build or pip install . to use package metadata")
    endif()

    target_compile_definitions(_native PRIVATE
        CALMETRICS_ENGINE_VERSION="${SKBUILD_PROJECT_VERSION}"
        CALMETRICS_ENGINE_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
        CALMETRICS_ENGINE_ARCH="${CMAKE_SYSTEM_PROCESSOR}"
    )

    install(
        TARGETS _native
        LIBRARY DESTINATION calmetrics_engine
        RUNTIME DESTINATION calmetrics_engine
    )
    list(APPEND calmetrics_engine_targets _native)
endif()

if(CALMETRICS_ENGINE_BUILD_TESTS)
    enable_testing()
    add_executable(calmetrics_engine_native_tests tests/native_tests.cpp)
    target_link_libraries(calmetrics_engine_native_tests PRIVATE calmetrics_engine_finance)
    add_test(NAME calmetrics_engine_native_tests COMMAND calmetrics_engine_native_tests)
    list(APPEND calmetrics_engine_targets calmetrics_engine_native_tests)
endif()

foreach(target IN LISTS calmetrics_engine_targets)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /EHsc /fp:precise /utf-8)
        target_compile_definitions(${target} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
    endif()

    if(CALMETRICS_ENGINE_SANITIZE)
        if(MSVC)
            message(FATAL_ERROR "CALMETRICS_ENGINE_SANITIZE currently requires GCC or Clang")
        endif()
        target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
endforeach()
