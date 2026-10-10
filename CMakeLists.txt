cmake_minimum_required(VERSION 3.20...4.2)

if(CMAKE_HOST_APPLE AND NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET AND "$ENV{MACOSX_DEPLOYMENT_TARGET}" STREQUAL "")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum supported macOS")
endif()

project(calmetrics_engine LANGUAGES CXX)

option(CALMETRICS_ENGINE_BUILD_PYTHON "Build the Python extension" ON)
option(CALMETRICS_ENGINE_BUILD_TESTS "Build standalone native tests" OFF)
option(CALMETRICS_ENGINE_BUILD_BENCHMARKS "Build local benchmark executables" OFF)
option(CALMETRICS_ENGINE_SANITIZE "Enable address/undefined-behavior sanitizers" OFF)
option(CALMETRICS_ENGINE_USE_EIGEN "Use eligible header-only Eigen numerical kernels" ON)
option(CALMETRICS_ENGINE_USE_ACCELERATE "Use guarded Apple Accelerate numerical kernels" ON)
option(CALMETRICS_ENGINE_USE_HIGHWAY "Use eligible Highway order selection kernels" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

find_package(Threads REQUIRED)
set(calmetrics_engine_arch "${CMAKE_SYSTEM_PROCESSOR}")
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    set(calmetrics_engine_arch "${CMAKE_OSX_ARCHITECTURES}")
endif()

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
    cpp/operators/recurrence.cpp
    cpp/operators/state_events.cpp
    cpp/operators/matrix.cpp
    cpp/operators/matrix_backend.cpp
    cpp/operators/covchol_accelerate.cpp
    cpp/operators/solve_accelerate.cpp
    cpp/operators/matmul_packed.cpp
    cpp/operators/matmul_accelerate.cpp
    cpp/operators/covariance_accelerate.cpp
    cpp/operators/transcendental_accelerate.cpp
    cpp/operators/matrix_foundation.cpp
    cpp/operators/matmul_statistics.cpp
    cpp/operators/axis_primitives.cpp
    cpp/operators/numeric_line_sort.cpp
    cpp/operators/latent.cpp
    cpp/operators/state.cpp
    cpp/operators/simd.cpp
    cpp/result_layout.cpp
    cpp/graph.cpp
    cpp/affine_projection.cpp
    cpp/compact_path.cpp
    cpp/pointwise.cpp
    cpp/compiler.cpp
    cpp/excel.cpp
    cpp/excel_budget.cpp
    cpp/excel_operators.cpp
    cpp/excel_state.cpp
    cpp/excel_matrix_foundation.cpp
    cpp/excel_axis_primitives.cpp
    cpp/excel_latent.cpp
    cpp/excel_scopes.cpp
    cpp/typed_ir.cpp
    cpp/typed_foundation.cpp
    cpp/typed_latent.cpp
    cpp/planner.cpp
    cpp/shared_memory.cpp
    cpp/model_payload.cpp
    cpp/native_runtime.cpp
    cpp/native_process.cpp
)
# The pinned internal Eigen specialization is confined to one translation unit.
set_source_files_properties(cpp/operators/matmul_packed.cpp PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
if(NOT MSVC)
    set_source_files_properties(cpp/operators/matmul_packed.cpp PROPERTIES COMPILE_OPTIONS "-fno-lto")
endif()
target_include_directories(calmetrics_engine_operators PUBLIC cpp/include)
target_link_libraries(calmetrics_engine_operators PUBLIC calmetrics_engine_scheduler)
if(APPLE AND calmetrics_engine_arch MATCHES "^(arm64|aarch64)$" AND CALMETRICS_ENGINE_USE_ACCELERATE)
    find_library(CALMETRICS_ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    target_link_libraries(calmetrics_engine_operators PUBLIC "${CALMETRICS_ACCELERATE_FRAMEWORK}")
    target_compile_definitions(calmetrics_engine_operators PRIVATE CALMETRICS_HAVE_ACCELERATE_COVCHOL=1)
endif()
if(CALMETRICS_ENGINE_USE_EIGEN)
    # Keep offline sdist builds reproducible. The unchanged upstream archive is
    # verified before extraction; no package import/request downloads or builds.
    set(calmetrics_eigen_archive "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen/eigen-3.4.0.tar.gz")
    set(calmetrics_eigen_sha "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72")
    file(SHA256 "${calmetrics_eigen_archive}" calmetrics_eigen_actual_sha)
    if(NOT calmetrics_eigen_actual_sha STREQUAL calmetrics_eigen_sha)
        message(FATAL_ERROR "Bundled Eigen archive checksum mismatch")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${calmetrics_eigen_archive}")
    set(calmetrics_eigen_root "${CMAKE_CURRENT_BINARY_DIR}/_deps/eigen-3.4.0")
    file(ARCHIVE_EXTRACT INPUT "${calmetrics_eigen_archive}" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/_deps")
    target_include_directories(calmetrics_engine_operators SYSTEM PRIVATE "${calmetrics_eigen_root}")
    target_compile_definitions(calmetrics_engine_operators PRIVATE CALMETRICS_HAVE_EIGEN_ROTATION=1)
endif()
install(FILES third_party/eigen/COPYING.MPL2 third_party/eigen/COPYING.BSD
    third_party/eigen/COPYING.README third_party/eigen/README.md
    DESTINATION calmetrics_engine/licenses/eigen)
if(APPLE AND calmetrics_engine_arch MATCHES "^(arm64|aarch64)$" AND CALMETRICS_ENGINE_USE_HIGHWAY)
    # Adopt only the measured platform; other builds retain native selection.
    set(calmetrics_highway_archive "${CMAKE_CURRENT_SOURCE_DIR}/third_party/highway/highway-1.3.0.tar.gz")
    file(SHA256 "${calmetrics_highway_archive}" calmetrics_highway_sha)
    if(NOT calmetrics_highway_sha STREQUAL "07b3c1ba2c1096878a85a31a5b9b3757427af963b1141ca904db2f9f4afe0bc2")
        message(FATAL_ERROR "Bundled Highway archive checksum mismatch")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${calmetrics_highway_archive}")
    set(calmetrics_highway_root "${CMAKE_CURRENT_BINARY_DIR}/_deps/highway-1.3.0")
    file(ARCHIVE_EXTRACT INPUT "${calmetrics_highway_archive}" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/_deps")
    set(HWY_FORCE_STATIC_LIBS ON CACHE BOOL "" FORCE)
    set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
    set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(HWY_CMAKE_HEADER_ONLY OFF CACHE BOOL "" FORCE)
    add_subdirectory("${calmetrics_highway_root}" "${CMAKE_CURRENT_BINARY_DIR}/_deps/highway-build" EXCLUDE_FROM_ALL)
    add_library(calmetrics_engine_highway_order STATIC
        cpp/operators/mask_validation_highway.cpp
        "${calmetrics_highway_root}/hwy/contrib/sort/vqsort.cc"
        "${calmetrics_highway_root}/hwy/contrib/sort/vqsort_f64a.cc")
    target_link_libraries(calmetrics_engine_highway_order PUBLIC hwy)
    target_include_directories(calmetrics_engine_highway_order PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/cpp/operators")
    target_compile_options(calmetrics_engine_highway_order PRIVATE $<TARGET_PROPERTY:hwy,COMPILE_OPTIONS>)
    target_link_libraries(calmetrics_engine_operators PRIVATE calmetrics_engine_highway_order)
    target_compile_definitions(calmetrics_engine_operators PRIVATE CALMETRICS_HAVE_HIGHWAY_ORDER=1)
endif()
install(FILES third_party/highway/LICENSE third_party/highway/LICENSE-BSD3 third_party/highway/README.md
    DESTINATION calmetrics_engine/licenses/highway)
if(UNIX AND NOT APPLE)
    target_link_libraries(calmetrics_engine_operators PUBLIC rt)
endif()

# CPU-specific flags apply only to this translation unit. Baseline code and module
# initialization remain safe on CPUs without AVX2. The runtime checks OS YMM support.
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
if(NOT MSVC)
    # Register fusion must preserve separate multiply/add rounding, including
    # on arm64 targets where the baseline ISA already has FMA instructions.
    set_property(SOURCE cpp/operators/simd.cpp cpp/operators/simd_avx2.cpp
        APPEND PROPERTY COMPILE_OPTIONS "-ffp-contract=off")
endif()
target_link_libraries(calmetrics_worker PRIVATE calmetrics_engine_operators)
install(TARGETS calmetrics_worker RUNTIME DESTINATION calmetrics_engine)
set(calmetrics_engine_targets calmetrics_engine_scheduler calmetrics_engine_finance calmetrics_engine_operators calmetrics_worker)
if(TARGET calmetrics_engine_highway_order)
    list(APPEND calmetrics_engine_targets hwy calmetrics_engine_highway_order)
endif()

if(CALMETRICS_ENGINE_BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_native cpp/bindings.cpp cpp/operator_bindings.cpp cpp/graph_bindings.cpp cpp/native_api_bindings.cpp cpp/excel_bindings.cpp)
    target_link_libraries(_native PRIVATE calmetrics_engine_finance calmetrics_engine_operators)

    if(NOT DEFINED SKBUILD_PROJECT_VERSION)
        message(FATAL_ERROR "Build Python via python -m build or pip install . to use package metadata")
    endif()

    # Identity includes the actual source tree (including uncommitted edits), toolchain and flags.
    file(GLOB_RECURSE calmetrics_identity_files CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/*")
    list(APPEND calmetrics_identity_files "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt" "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml")
    list(SORT calmetrics_identity_files)
    set(calmetrics_identity "${SKBUILD_PROJECT_VERSION};${CMAKE_CXX_COMPILER_ID};${CMAKE_CXX_COMPILER_VERSION};${calmetrics_engine_arch};${CMAKE_BUILD_TYPE};${CMAKE_CXX_FLAGS};${CMAKE_CXX_FLAGS_RELEASE}")
    string(APPEND calmetrics_identity ";eigen_rotation=${CALMETRICS_ENGINE_USE_EIGEN}")
    string(APPEND calmetrics_identity ";accelerate_covchol=${CALMETRICS_ENGINE_USE_ACCELERATE}")
    string(APPEND calmetrics_identity ";highway_order=${CALMETRICS_ENGINE_USE_HIGHWAY}")
    foreach(source IN LISTS calmetrics_identity_files)
        file(SHA256 "${source}" source_hash)
        file(RELATIVE_PATH source_name "${CMAKE_CURRENT_SOURCE_DIR}" "${source}")
        string(APPEND calmetrics_identity ";${source_name}:${source_hash}")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source}")
    endforeach()
    string(SHA256 calmetrics_build_id "${calmetrics_identity}")
    target_compile_definitions(_native PRIVATE
        CALMETRICS_ENGINE_BUILD_ID="${calmetrics_build_id}"
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
    add_executable(calmetrics_engine_excel_tests tests/excel_native_tests.cpp)
    target_link_libraries(calmetrics_engine_excel_tests PRIVATE calmetrics_engine_operators)
    add_test(NAME calmetrics_engine_excel_tests COMMAND calmetrics_engine_excel_tests)
    list(APPEND calmetrics_engine_targets calmetrics_engine_excel_tests)
    add_executable(calmetrics_engine_native_tests tests/native_tests.cpp)
    target_link_libraries(calmetrics_engine_native_tests PRIVATE calmetrics_engine_finance)
    add_test(NAME calmetrics_engine_native_tests COMMAND calmetrics_engine_native_tests)
    # Fail a blocked admission regression instead of hanging CI indefinitely.
    set_tests_properties(calmetrics_engine_native_tests PROPERTIES TIMEOUT 60)
    add_executable(calmetrics_engine_operator_tests tests/operator_native_tests.cpp)
    target_link_libraries(calmetrics_engine_operator_tests PRIVATE calmetrics_engine_operators)
    if(APPLE AND calmetrics_engine_arch MATCHES "^(arm64|aarch64)$" AND CALMETRICS_ENGINE_USE_ACCELERATE)
        target_compile_definitions(calmetrics_engine_operator_tests PRIVATE CALMETRICS_EXPECT_ACCELERATE_THREADING=1)
    endif()
    if(CALMETRICS_ENGINE_USE_EIGEN)
        target_compile_definitions(calmetrics_engine_operator_tests PRIVATE CALMETRICS_EXPECT_EIGEN_INPUT_SCAN=1)
    else()
        target_compile_definitions(calmetrics_engine_operator_tests PRIVATE CALMETRICS_EXPECT_EIGEN_INPUT_SCAN=0)
    endif()
    if(TARGET calmetrics_engine_highway_order)
        target_compile_definitions(calmetrics_engine_operator_tests PRIVATE CALMETRICS_EXPECT_HIGHWAY_ORDER=1)
    else()
        target_compile_definitions(calmetrics_engine_operator_tests PRIVATE CALMETRICS_EXPECT_HIGHWAY_ORDER=0)
    endif()
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
    add_executable(calmetrics_engine_math_tests tests/math_native_tests.cpp)
    target_link_libraries(calmetrics_engine_math_tests PRIVATE calmetrics_engine_operators)
    add_test(NAME calmetrics_engine_math_tests COMMAND calmetrics_engine_math_tests)
    add_executable(calmetrics_engine_stateful_tests tests/stateful_native_tests.cpp)
    target_link_libraries(calmetrics_engine_stateful_tests PRIVATE calmetrics_engine_operators)
    add_test(NAME calmetrics_engine_stateful_tests COMMAND calmetrics_engine_stateful_tests)
    add_executable(calmetrics_engine_result_tests tests/result_native_tests.cpp)
    target_link_libraries(calmetrics_engine_result_tests PRIVATE calmetrics_engine_operators)
    add_test(NAME calmetrics_engine_result_tests COMMAND calmetrics_engine_result_tests $<TARGET_FILE:calmetrics_worker>)
    set_tests_properties(calmetrics_engine_result_tests PROPERTIES TIMEOUT 60)
    list(APPEND calmetrics_engine_targets
        calmetrics_engine_result_tests
        calmetrics_engine_stateful_tests
        calmetrics_engine_math_tests
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
