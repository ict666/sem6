.PHONY: all clean

all: task

task: task.cpp
	g++ -O2 -fopenmp -o task task.cpp

polus: task.cpp
	xlc++_r -O2 -qsmp=omp -o task task.cpp

mac: task.cpp
	clang++ -Xpreprocessor -fopenmp task.cpp -o task -lomp