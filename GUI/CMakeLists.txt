cmake_minimum_required(VERSION 3.24)
project(BarnesHutSimulation LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Include FetchContent for external dependencies
include(FetchContent)

# GLFW
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
)

# GLM (OpenGL Mathematics)
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
)

# Make dependencies available
FetchContent_MakeAvailable(glfw glm)

# Locate CUDA kernel source files
file(GLOB CUDA_KERNEL_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ui/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sfc/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

# Add executable
add_executable(BarnesHutSimulation 
    src/main.cpp
    ${CUDA_KERNEL_SOURCES}
)

# CUDA Configuration
set_target_properties(BarnesHutSimulation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)

# Include directories
target_include_directories(BarnesHutSimulation PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ui  # Nuevo directorio
    ${OpenCV_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(BarnesHutSimulation 
    ${OpenCV_LIBS}
    OpenGL::GL
    GLEW::GLEW
    glfw
    glm
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