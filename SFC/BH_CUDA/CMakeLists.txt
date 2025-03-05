cmake_minimum_required(VERSION 3.24)
project(main LANGUAGES CXX CUDA)
add_executable(${PROJECT_NAME})

target_sources(${PROJECT_NAME} PRIVATE
    main.cu
    barnes_hut.cu
)

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found")
endif()
include_directories(${OpenCV_INCLUDE_DIRS})

set_target_properties(${PROJECT_NAME} PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CMAKE_CXX_STANDARD 20
)

find_package(OpenGL REQUIRED)
find_package(OpenMP REQUIRED)

# Link GLFW and OpenGL libraries
target_link_libraries(${PROJECT_NAME} PUBLIC
    OpenGL::GL
    OpenMP::OpenMP_CXX
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)