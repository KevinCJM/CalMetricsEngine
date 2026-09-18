cmake_minimum_required(VERSION 3.20...4.2)
# Set the deployment target before project() initializes the compiler.
if(CMAKE_HOST_APPLE AND NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET AND "$ENV{MACOSX_DEPLOYMENT_TARGET}" STREQUAL "")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum supported macOS")
endif()
project(my_ctools LANGUAGES CXX)

option(MY_CTOOLS_BUILD_PYTHON "Build the Python extension" ON)
option(MY_CTOOLS_BUILD_TESTS "Build standalone native tests" OFF)
option(MY_CTOOLS_SANITIZE "Enable address/undefined-behavior sanitizers" OFF)
# No host-specific ISA flags: published wheels must run on baseline CPUs.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
find_package(Threads REQUIRED)

add_library(my_ctools_kernels STATIC
    cpp/kernels/statistics.cpp
    cpp/kernels/drawdown.cpp
    cpp/kernels/streaks.cpp
    cpp/kernels/rolling.cpp
)
target_include_directories(my_ctools_kernels PUBLIC cpp/include)
target_link_libraries(my_ctools_kernels PUBLIC Threads::Threads)
set(my_ctools_targets my_ctools_kernels)

if(MY_CTOOLS_BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_core cpp/bindings.cpp)
    target_link_libraries(_core PRIVATE my_ctools_kernels)
    if(NOT DEFINED SKBUILD_PROJECT_VERSION)
        message(FATAL_ERROR "Build Python via python -m build or pip install . to use package metadata")
    endif()
    target_compile_definitions(_core PRIVATE
        MY_CTOOLS_VERSION="${SKBUILD_PROJECT_VERSION}"
        MY_CTOOLS_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
        MY_CTOOLS_ARCH="${CMAKE_SYSTEM_PROCESSOR}"
    )
    install(TARGETS _core LIBRARY DESTINATION my_ctools RUNTIME DESTINATION my_ctools)
    list(APPEND my_ctools_targets _core)
endif()

if(MY_CTOOLS_BUILD_TESTS)
    enable_testing()
    add_executable(my_ctools_native_tests tests/native_tests.cpp)
    target_link_libraries(my_ctools_native_tests PRIVATE my_ctools_kernels)
    add_test(NAME my_ctools_native_tests COMMAND my_ctools_native_tests)
    list(APPEND my_ctools_targets my_ctools_native_tests)
endif()

foreach(target IN LISTS my_ctools_targets)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /EHsc /fp:precise /utf-8)
        target_compile_definitions(${target} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
    endif()
    if(MY_CTOOLS_SANITIZE)
        if(MSVC)
            message(FATAL_ERROR "MY_CTOOLS_SANITIZE currently requires GCC or Clang")
        endif()
        target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
endforeach()
