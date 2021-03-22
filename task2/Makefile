.PHONY: all clean

all: task

task: task.cpp
	mpic++ -o task task.cpp

polus: task.cpp
	mpixlC -std=c++11 -o task2 task2.cpp

test_pc: task
	#1
	mpirun -n 1 task 16 1 0 1
	#2
	mpirun -n 2 task 16 1 1 0 test
	#4
	mpirun -n 4 task 16 1 1 0 test
	#8
	mpirun -n 8 task 16 1 1 0 test

test_mac: task
	#1
	mpirun -n 1 task 16 1 0 1
	#2
	mpirun -n 2 task 16 1 1 0 test
	#4
	mpirun -n 4 task 16 1 1 0 test
