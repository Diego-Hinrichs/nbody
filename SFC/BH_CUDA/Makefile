# Compiler settings
NVCC = nvcc
CXX = g++
CXXFLAGS = -std=c++20
NVCC_FLAGS = -std=c++20

# OpenCV settings (assuming pkg-config is available)
OPENCV_FLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS = $(shell pkg-config --libs opencv4)

# OpenMP flags for host compiler
OPENMP_FLAGS = -Xcompiler -fopenmp

# CUDA paths and flags
CUDA_INCLUDE_PATH = /usr/local/cuda/include
CUDA_LIBRARY_PATH = /usr/local/cuda/lib64
CUDA_LIBS = -lcudart -lcuda

# Include paths
INCLUDES = -I$(CUDA_INCLUDE_PATH) $(OPENCV_FLAGS)

# Libraries
LIBS = -L$(CUDA_LIBRARY_PATH) $(CUDA_LIBS) -lGL $(OPENCV_LIBS) $(OPENMP_FLAGS)

# Source files
SOURCES = main.cu barnesHutCuda.cu barnesHut_kernel.cu

# Object files
OBJECTS = $(SOURCES:.cu=.o)

# Executable name
EXECUTABLE = main

# Phony targets
.PHONY: all clean

# Default target
all: $(EXECUTABLE)

# Linking - Changed to use the device-link option first
$(EXECUTABLE): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -dlink $(OBJECTS) -o device_link.o
	$(NVCC) $(NVCC_FLAGS) $(OBJECTS) device_link.o -o $@ $(LIBS)

# Compilation pattern rule for CUDA files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -dc $< -o $@

# Clean
clean:
	rm -f $(OBJECTS) $(EXECUTABLE) device_link.o