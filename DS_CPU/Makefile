CXX      = g++
TARGET   = main
SRC      = main.cpp
CXXFLAGS = -fopenmp `pkg-config --cflags opencv4`
LDFLAGS  = -fopenmp `pkg-config --libs opencv4`

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(SRC) -o $(TARGET) $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
