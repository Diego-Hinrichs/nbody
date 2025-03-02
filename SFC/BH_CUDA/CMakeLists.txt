cmake_minimum_required(VERSION 3.24)
project(main LANGUAGES CXX CUDA)
include(FetchContent)
add_executable(${PROJECT_NAME})

# Add source files
# file(GLOB SOURCES *.cpp *.h *.cu *.cuh)
target_sources(${PROJECT_NAME} PRIVATE
    main.cu
    barnesHutCuda.cu
    barnesHut_kernel.cu
)

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found")
endif()
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable
# cuda_add_executable(BarnesHut ${SOURCES})
set_target_properties(${PROJECT_NAME} PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CMAKE_CXX_STANDARD 20
)

# GLFW
set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS    OFF)
set(GLFW_BUILD_DOCS     OFF)
set(GLFW_BUILD_WAYLAND  OFF) # Do not include Wayland support in GLFW
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        7b6aead9fb88b3623e3b3725ebb42670cbe4c579 # 3.4
    GIT_SHALLOW    ON
)

# glm
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        0af55ccecd98d4e5a8d1fad7de25ba429d60e863 # 1.0.1
    GIT_SHALLOW    ON
    FIND_PACKAGE_ARGS
)

FetchContent_MakeAvailable(glfw glm)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenMP REQUIRED)

# Link GLFW and OpenGL libraries
target_link_libraries(${PROJECT_NAME} PUBLIC
    OpenGL::GL
    GLEW
    glm
    glfw
    OpenMP::OpenMP_CXX
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)