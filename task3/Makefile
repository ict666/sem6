.PHONY: all clean

all: task

task: task.cpp
	mpic++ -std=c++11 -O2 -o task task.cpp

bluegene: task.cpp
	mpixlcxx_r -O2 -o task task.cpp

time: bluegene
	for n in 1 2 4 8 16 32 64 128 256 ; do \
		mpisubmit.bg -n $$n -w 00:10 task -- 28 0 1 ; \
	done

accuracy: bluegene
	for n in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do \
		mpisubmit.bg -n 32 -w 00:10 task -- 26 0 1 ; \
	done


