cmake_minimum_required(VERSION 3.20)

project(TurboQuantQuantization C CXX)

# ── 1. Compiler settings ──────────────────────────────────────────────────────
# Use Intel icx if available, otherwise system default (gcc/clang).
# Override at configure time: cmake -DCMAKE_C_COMPILER=icx ..
find_program(ICX_PATH icx)
if(ICX_PATH)
    set(CMAKE_C_COMPILER    icx)
    set(CMAKE_CXX_COMPILER  icpx)
else()
    message(STATUS "Intel icx not found; using system default compiler.")
endif()

set(CMAKE_C_STANDARD    11)
set(CMAKE_CXX_STANDARD  17)

# ── 2. Sources ────────────────────────────────────────────────────────────────
set(TURBOQUANT_SOURCES
    src/turboquant.c
    src/lin_alg.c
    src/codebook.c
)

# ── 3. Shared library target ──────────────────────────────────────────────────
add_library(turboquant SHARED ${TURBOQUANT_SOURCES})

# ── 4. Compiler flags (mirrors your icx invocation) ──────────────────────────
target_compile_options(turboquant PRIVATE
    -mavx2
    -mfma
    -march=native
    $<$<CONFIG:Debug>:   -g -O0>
    $<$<CONFIG:Release>: -O3>
)

# ── 5. Intel oneAPI include paths ─────────────────────────────────────────────
# Adjust ONEAPI_ROOT if your installation differs
set(ONEAPI_ROOT "/opt/intel/oneapi" CACHE PATH "Intel oneAPI root directory")

target_include_directories(turboquant PRIVATE
    ${ONEAPI_ROOT}/compiler/2026.0/lib/clang/22/include
)

# ── 6. Link Intel runtime + stdc++ (matches your -lstdc++) ───────────────────
target_link_directories(turboquant PRIVATE
    ${ONEAPI_ROOT}/compiler/latest/lib
)

target_link_libraries(turboquant PRIVATE
    stdc++
    # Uncomment what you actually use:
    # mkl_intel_lp64 mkl_sequential mkl_core   # MKL
    svml                                       # Intel short vector math lib
    # irc                                        # Intel runtime
)

# ── 7. Find OpenMP for multi-threaded batch API ────────────────────────────
find_package(OpenMP)
if(OpenMP_C_FOUND)
    # If using Intel compiler, ensure we use -qopenmp
    if(CMAKE_C_COMPILER_ID STREQUAL "IntelLLVM")
        target_compile_options(turboquant PRIVATE -qopenmp)
        target_link_options(turboquant PRIVATE -qopenmp)
    else()
        target_link_libraries(turboquant PRIVATE OpenMP::OpenMP_C)
    endif()
else()
    message(WARNING "OpenMP not found: multi-threaded batch API will not build.")
endif()

# ── 8. Output goes to build/ (mirrors your -o build/...) ─────────────────────
set_target_properties(turboquant PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build
    OUTPUT_NAME              turboquant        # → libturboquant.so
    POSITION_INDEPENDENT_CODE ON              # mandatory for shared libs
)

