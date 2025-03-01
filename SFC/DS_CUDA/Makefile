# Makefile para compilar la implementación CUDA con OpenCV y ordenamiento SFC

NVCC      = nvcc
TARGET    = main
SRC       = main.cu
NVCCFLAGS = -O2 `pkg-config --cflags opencv4`
LDFLAGS   = `pkg-config --libs opencv4`

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) -o $(TARGET) $(NVCCFLAGS) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
