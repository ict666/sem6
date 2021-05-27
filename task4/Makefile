test: lib.h lib.cpp generate.h
	mpic++ -g -o main lib.cpp -lm
	mpirun -np 1 main Canonization
	mpirun -np 2 main Canonization
	mpirun -np 4 main Canonization
	mpirun -np 1 main Black_box n 4 k 2 l 3
	mpirun -np 2 main Black_box n 4 k 2 l 3
	mpirun -np 4 main Black_box n 4 k 2 l 3
	mpirun -np 1 main Black_box n 2 k 1 l 2
	mpirun -np 2 main Black_box n 2 k 1 l 2
	mpirun -np 4 main Black_box n 2 k 1 l 2