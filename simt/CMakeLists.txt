cmake_minimum_required(VERSION 3.20)
project(TurboQuantQuantization LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(TURBOQUANT_BUILD_CUDA "Build CUDA targets (turboquant library + context factory)." ON)

# Host-only target that can be built without CUDA (useful on non-GPU dev machines).
add_library(turboquant_codebook STATIC
    src/codebook.c
)
target_include_directories(turboquant_codebook
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(turboquant_codebook
    PUBLIC
        m
)

if (TURBOQUANT_BUILD_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)

        add_library(turboquant SHARED
            src/codebook.c
            src/lin_alg.cu
            src/turboquant.cu
        )
        target_include_directories(turboquant
            PUBLIC
                ${PROJECT_SOURCE_DIR}/include
        )
        set_target_properties(turboquant PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            POSITION_INDEPENDENT_CODE ON
        )
        target_link_libraries(turboquant
            PUBLIC
                CUDA::cudart
                CUDA::cublas
                CUDA::cusolver
                CUDA::curand
                m
        )

        add_executable(turboquant_context_factory
            src/initialize_context.c
        )
        target_include_directories(turboquant_context_factory
            PRIVATE
                ${PROJECT_SOURCE_DIR}/include
        )
        target_link_libraries(turboquant_context_factory
            PRIVATE
                turboquant
        )
    else()
        message(WARNING "CUDA compiler not found: skipping CUDA targets. Install NVCC (e.g., in Google Colab) to build turboquant.")
    endif()
endif()
