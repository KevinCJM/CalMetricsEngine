cmake_minimum_required(VERSION 3.20...4.2)

if(CMAKE_HOST_APPLE AND NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET AND "$ENV{MACOSX_DEPLOYMENT_TARGET}" STREQUAL "")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum supported macOS")
endif()

project(calmetrics_engine LANGUAGES CXX)

option(CALMETRICS_ENGINE_BUILD_PYTHON "Build the Python extension" ON)
option(CALMETRICS_ENGINE_BUILD_TESTS "Build standalone native tests" OFF)
option(CALMETRICS_ENGINE_BUILD_BENCHMARKS "Build local benchmark executables" OFF)
option(CALMETRICS_ENGINE_SANITIZE "Enable address/undefined-behavior sanitizers" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

find_package(Threads REQUIRED)

add_library(calmetrics_engine_scheduler STATIC
    cpp/scheduler.cpp
)
target_include_directories(calmetrics_engine_scheduler PUBLIC cpp/include)
target_link_libraries(calmetrics_engine_scheduler PUBLIC Threads::Threads)

add_library(calmetrics_engine_finance STATIC
    cpp/finance/statistics.cpp
    cpp/finance/drawdown.cpp
    cpp/finance/streaks.cpp
    cpp/finance/rolling.cpp
)
target_include_directories(calmetrics_engine_finance PUBLIC cpp/include)
target_link_libraries(calmetrics_engine_finance PUBLIC calmetrics_engine_scheduler)

add_library(calmetrics_engine_operators STATIC
    cpp/operators/registry.cpp
    cpp/operators/elementwise.cpp
    cpp/operators/reduction.cpp
    cpp/operators/sequence.cpp
    cpp/operators/matrix.cpp
    cpp/operators/state.cpp
    cpp/operators/simd.cpp
    cpp/graph.cpp
    cpp/compiler.cpp
    cpp/planner.cpp
    cpp/shared_memory.cpp
    cpp/native_runtime.cpp
    cpp/native_process.cpp
)
target_include_directories(calmetrics_engine_operators PUBLIC cpp/include)
target_link_libraries(calmetrics_engine_operators PUBLIC calmetrics_engine_scheduler)
if(UNIX AND NOT APPLE)
    target_link_libraries(calmetrics_engine_operators PUBLIC rt)
endif()

# CPU-specific flags apply only to this translation unit. Baseline code and module
# initialization remain safe on CPUs without AVX2. The runtime checks OS YMM support.
set(calmetrics_engine_arch "${CMAKE_SYSTEM_PROCESSOR}")
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    set(calmetrics_engine_arch "${CMAKE_OSX_ARCHITECTURES}")
endif()
if(calmetrics_engine_arch MATCHES "^(x86_64|AMD64|amd64)$")
    target_sources(calmetrics_engine_operators PRIVATE cpp/operators/simd_avx2.cpp)
    target_compile_definitions(calmetrics_engine_operators PRIVATE CALMETRICS_HAVE_AVX2=1)
    if(MSVC)
        set_source_files_properties(cpp/operators/simd_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    else()
        set_source_files_properties(cpp/operators/simd_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
    endif()
endif()

add_executable(calmetrics_worker cpp/native_worker_main.cpp)
target_link_libraries(calmetrics_worker PRIVATE calmetrics_engine_operators)
install(TARGETS calmetrics_worker RUNTIME DESTINATION calmetrics_engine)
set(calmetrics_engine_targets calmetrics_engine_scheduler calmetrics_engine_finance calmetrics_engine_operators calmetrics_worker)

if(CALMETRICS_ENGINE_BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_native cpp/bindings.cpp cpp/operator_bindings.cpp cpp/graph_bindings.cpp cpp/native_api_bindings.cpp)
    target_link_libraries(_native PRIVATE calmetrics_engine_finance calmetrics_engine_operators)

    if(NOT DEFINED SKBUILD_PROJECT_VERSION)
        message(FATAL_ERROR "Build Python via python -m build or pip install . to use package metadata")
    endif()

    target_compile_definitions(_native PRIVATE
        CALMETRICS_ENGINE_VERSION="${SKBUILD_PROJECT_VERSION}"
        CALMETRICS_ENGINE_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
        CALMETRICS_ENGINE_ARCH="${calmetrics_engine_arch}"
    )

    install(
        TARGETS _native
        LIBRARY DESTINATION calmetrics_engine
        RUNTIME DESTINATION calmetrics_engine
    )
    list(APPEND calmetrics_engine_targets _native)
endif()

if(CALMETRICS_ENGINE_BUILD_BENCHMARKS)
    add_executable(calmetrics_engine_multiworkload_benchmark tools/benchmark_multiworkload.cpp)
    target_link_libraries(calmetrics_engine_multiworkload_benchmark PRIVATE calmetrics_engine_operators)
    list(APPEND calmetrics_engine_targets calmetrics_engine_multiworkload_benchmark)
endif()

if(CALMETRICS_ENGINE_BUILD_TESTS)
    enable_testing()
    add_executable(calmetrics_engine_native_tests tests/native_tests.cpp)
    target_link_libraries(calmetrics_engine_native_tests PRIVATE calmetrics_engine_finance)
    add_test(NAME calmetrics_engine_native_tests COMMAND calmetrics_engine_native_tests)
    add_executable(calmetrics_engine_operator_tests tests/operator_native_tests.cpp)
    target_link_libraries(calmetrics_engine_operator_tests PRIVATE calmetrics_engine_operators)
    add_test(NAME calmetrics_engine_operator_tests COMMAND calmetrics_engine_operator_tests)
    add_executable(calmetrics_engine_graph_tests tests/graph_native_tests.cpp)
    target_link_libraries(calmetrics_engine_graph_tests PRIVATE calmetrics_engine_operators)
    add_test(NAME calmetrics_engine_graph_tests COMMAND calmetrics_engine_graph_tests)
    add_executable(calmetrics_engine_compiler_tests tests/compiler_native_tests.cpp)
    target_link_libraries(calmetrics_engine_compiler_tests PRIVATE calmetrics_engine_operators)
    add_test(NAME calmetrics_engine_compiler_tests COMMAND calmetrics_engine_compiler_tests)
    add_executable(calmetrics_engine_runtime_tests tests/runtime_native_tests.cpp)
    target_link_libraries(calmetrics_engine_runtime_tests PRIVATE calmetrics_engine_operators)
    add_test(NAME calmetrics_engine_runtime_tests COMMAND calmetrics_engine_runtime_tests $<TARGET_FILE:calmetrics_worker>)
    list(APPEND calmetrics_engine_targets
        calmetrics_engine_runtime_tests
        calmetrics_engine_compiler_tests
        calmetrics_engine_native_tests
        calmetrics_engine_operator_tests
        calmetrics_engine_graph_tests
    )
endif()

foreach(target IN LISTS calmetrics_engine_targets)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /EHsc /fp:precise /utf-8)
        target_compile_definitions(${target} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _WIN32_WINNT=0x0601)
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
