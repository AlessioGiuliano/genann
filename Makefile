CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wshadow -O3 -g -march=native
LXXFLAGS = -std=c++17
LDLIBS = -lm -fopenmp -L ${CUDA_PATH}lib64 -lcuda -lcudart -lstdc++

all: genann_cu.o example1 example2 example3 example4

sigmoid: CFLAGS += -Dgenann_act=genann_act_sigmoid_cached
sigmoid: CXXFLAGS += -Dgenann_act=genann_act_sigmoid_cached
sigmoid: all

threshold: CFLAGS += -Dgenann_act=genann_act_threshold
threshold: CXXFLAGS += -Dgenann_act=genann_act_threshold
threshold: all

linear: CFLAGS += -Dgenann_act=genann_act_linear
linear: CXXFLAGS += -Dgenann_act=genann_act_linear
linear: all

CUDAFLAGS = -arch=sm_60
genann_cu.o: genann.cu
	nvcc $(CUDAFLAGS) -c genann.cu -o genann_cu.o

example1: example1.o genann.o genann_cu.o

example2: example2.o genann.o genann_cu.o

example3: example3.o genann.o genann_cu.o

example4: example4.o genann.o genann_cu.o


clean:
	$(RM) *.o
	$(RM) example1 example2 example3 example4 *.exe
	$(RM) persist.txt

.PHONY: sigmoid threshold linear clean
