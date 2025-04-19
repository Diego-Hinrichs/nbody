cmake_minimum_required(VERSION 3.5)

set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc-13)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin=/usr/bin/gcc-13")

project(BarnesHutSimulation LANGUAGES C CXX CUDA)

# Set C and C++ compilers explicitly if needed
if(NOT DEFINED CMAKE_C_COMPILER)
  set(CMAKE_C_COMPILER gcc)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER g++)
endif()

# Ensure compiler rules are set
include(CMakeForceCompiler)
enable_language(C)
enable_language(CXX)
enable_language(CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find system packages
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Locate source files
file(GLOB SOURCES_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sfc/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/implementations/gpu/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/implementations/cpu/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/base/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ui/simulation_state.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cpp
)

# Define the executable sources
set(EXECUTABLE_SOURCES 
    src/main.cpp
    ${SOURCES_FILES}
)

# Add executable
add_executable(BarnesHutSimulation 
    ${EXECUTABLE_SOURCES}
)

# CUDA Configuration
set_target_properties(BarnesHutSimulation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Include directories
target_include_directories(BarnesHutSimulation PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/include/simulation/base
    ${CMAKE_CURRENT_SOURCE_DIR}/include/simulation/implementations/cpu
    ${CMAKE_CURRENT_SOURCE_DIR}/include/simulation/implementations/gpu
    ${CMAKE_CURRENT_SOURCE_DIR}/external/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(BarnesHutSimulation 
    OpenMP::OpenMP_CXX
    CUDA::cudart
    CUDA::cuda_driver
)

# Optional: Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(BarnesHutSimulation PRIVATE 
        -O3 -march=native -mtune=native
    )
endif()

# CUDA-specific compilation options
target_compile_options(BarnesHutSimulation PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler -fopenmp
        --expt-extended-lambda
        --expt-relaxed-constexpr
        --relocatable-device-code=true
    >
)