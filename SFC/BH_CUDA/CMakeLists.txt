cmake_minimum_required(VERSION 3.24)
project(main LANGUAGES CXX CUDA)

# Opciones de compilación
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Configuraciones de CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress 611")

# Definir el ejecutable
add_executable(${PROJECT_NAME})

# Agregar fuentes
target_sources(${PROJECT_NAME} PRIVATE
    main.cu
    barnes_hut.cu
)

# Encontrar dependencias
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found")
endif()

find_package(OpenGL REQUIRED)
find_package(OpenMP REQUIRED)

# Incluir directorios
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Propiedades del target
set_target_properties(${PROJECT_NAME} PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Enlazar bibliotecas
target_link_libraries(${PROJECT_NAME} PUBLIC
    OpenGL::GL
    OpenMP::OpenMP_CXX
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)

# Información adicional
message(STATUS "CUDA version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")