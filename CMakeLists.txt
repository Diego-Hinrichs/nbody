cmake_minimum_required(VERSION 3.24)

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

project(BarnesHutSimulation LANGUAGES C CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find system packages
find_package(OpenMP REQUIRED)
find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Include FetchContent for external dependencies
include(FetchContent)

# Disable system GLFW if it exists
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)

# Explicitly unset any existing GLFW targets
unset(glfw_FOUND CACHE)
unset(GLFW_FOUND CACHE)

# Use FetchContent for all dependencies
FetchContent_Declare(
    glfw
    URL https://github.com/glfw/glfw/archive/refs/tags/3.4.zip
    OVERRIDE_FIND_PACKAGE
)

# GLM (OpenGL Mathematics)
FetchContent_Declare(
    glm
    URL https://github.com/g-truc/glm/archive/refs/tags/0.9.9.8.zip
)

# ImGui
FetchContent_Declare(
    imgui
    URL https://github.com/ocornut/imgui/archive/refs/tags/v1.91.9.zip
)


# Disable package finding for these libraries
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG TRUE)
set(CMAKE_FIND_PACKAGE_QUIET TRUE)

# Make dependencies available
FetchContent_MakeAvailable(glfw glm)

# Prepare ImGui manually
FetchContent_GetProperties(imgui)
if(NOT imgui_POPULATED)
    FetchContent_Populate(imgui)
endif()

# ImGui source files
set(IMGUI_SOURCES
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

# GLAD source and header files
set(GLAD_SOURCE 
    ${CMAKE_CURRENT_SOURCE_DIR}/external/glad/src/glad.c
)
set(GLAD_INCLUDE_DIR 
    ${CMAKE_CURRENT_SOURCE_DIR}/external/glad/include
)

# Verify GLAD header exists
if(NOT EXISTS ${GLAD_INCLUDE_DIR}/glad/glad.h)
    message(FATAL_ERROR "GLAD header not found in ${GLAD_INCLUDE_DIR}")
endif()

# Locate source files
file(GLOB SOURCES_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ui/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sfc/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/implementations/gpu/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/implementations/cpu/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/simulation/base/*.cu
)

# First, create a separate target for GLAD
add_library(glad STATIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/external/glad/src/glad.c
)

# Set include directories for GLAD
target_include_directories(glad PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/external/glad/include
)

# Define the executable sources without GLAD
set(EXECUTABLE_SOURCES 
    src/main.cpp
    ${SOURCES_FILES}
    ${IMGUI_SOURCES}
    src/ui/simulation_ui_manager.cpp
    src/ui/opengl_renderer.cpp
)

# Add executable
add_executable(BarnesHutSimulation 
    ${EXECUTABLE_SOURCES}
)

# CUDA Configuration
set_target_properties(BarnesHutSimulation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)

# Include directories
target_include_directories(BarnesHutSimulation PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ui
    ${CMAKE_CURRENT_SOURCE_DIR}/include/simulation/base
    ${CMAKE_CURRENT_SOURCE_DIR}/include/simulation/implementations/cpu
    ${CMAKE_CURRENT_SOURCE_DIR}/include/simulation/implementations/gpu
    ${CMAKE_CURRENT_SOURCE_DIR}/external/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}
    ${glfw_SOURCE_DIR}/include
    ${glm_SOURCE_DIR}
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
    ${GLAD_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(BarnesHutSimulation 
    glad
    OpenGL::GL
    OpenMP::OpenMP_CXX
    glfw
    glm
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