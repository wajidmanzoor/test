cmake_minimum_required(VERSION 3.10)
project(CDS LANGUAGES CXX CUDA)

# A100 uses SM 80
set(CMAKE_CUDA_ARCHITECTURES 80)

# Enable modern CUDA features
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

add_executable(main
    src/graph.cpp
    src/common.cpp
    src/gpuMemoryAllocation.cu
    src/helpers.cu
    main.cu
)

# Extended lambda and better default stream
target_compile_options(main PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>
)

target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Link CUDA libraries (if needed)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(main PRIVATE CUDA::cublas CUDA::cufft CUDA::curand)