CXX = g++
CXXFLAGS = -fopenmp -std=c++17 -Wall -Wshadow -O3 -g -march=native
LXXFLAGS = -std=c++17
LDLIBS = -lm -fopenmp -L ${CUDA_PATH}lib64 -lcuda -lcudart -lstdc++

# Default target
all: main_fast main_slow test

# Compile object files
genann_cu.o: genann.cu
	nvcc -c genann.cu -o genann_cu.o

example4.o: example4.cpp
	$(CXX) $(CXXFLAGS) -c example4.cpp -o example4.o

genann.o: genann.cpp
	$(CXX) $(CXXFLAGS) -c genann.cpp -o genann.o

genann_fast.o: genann.cpp
	$(CXX) $(CXXFLAGS) -c genann.cpp -o genann_fast.o -DFAST

# Build main_fast with accelerated definition
main_fast: example4.o genann_fast.o genann_cu.o
	$(CXX) $(CXXFLAGS) example4.o genann_fast.o genann_cu.o -o main_fast $(LDLIBS)

# Build main_slow without accelerated definition
main_slow: example4.o genann.o genann_cu.o
	$(CXX) $(CXXFLAGS) example4.o genann.o genann_cu.o -o main_slow $(LDLIBS)

test: test.o genann.o genann_cu.o
	$(CXX) $(CXXFLAGS) test.o genann_fast.o genann_cu.o -o test $(LDLIBS)

# Clean up build files
clean:
	$(RM) *.o
	$(RM) example4 *.exe
	$(RM) main_fast main_slow *.exe
	$(RM) persist.txt

.PHONY: clean

