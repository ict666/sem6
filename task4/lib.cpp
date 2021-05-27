#include "lib.h"
#include "generate.h"

void box_check(unsigned long long seg_size, complexd *V, std::string name, int rank, int size, int rc, MPI_Status &stat) {
	float fr_snd = 0;
    float ans = 0;
	for (int i = 0; i < seg_size; i++) {
        	fr_snd += abs(V[i] * V[i]);
        }
        if(rank == 0){
        	ans += fr_snd;
        	for (int i = 1; i < size; i++){
				MPI_Recv(&fr_snd, 1, MPI_DOUBLE, i, 9, MPI_COMM_WORLD, &stat);
				ans += fr_snd;
        	}
        	cout << name << ans << endl;
        }
        else{
        	rc = MPI_Send(&fr_snd, 1, MPI_DOUBLE, 0, 9, MPI_COMM_WORLD);
        }
}

void my_vec_1(complexd *V, int size, int rank) {
	if(size == 1){
    			V[0] = complexd(0, 0);
    			V[1] = complexd(1, 0);
    		}
	else{
		if(rank == 0){
			V[0] = complexd(0, 0);
		}
		else{
			V[0] = complexd(1, 0);
		}
	}
}

void my_vec_2(complexd *V, int size, int rank) {
	if(size == 1){
    		V[0] = complexd(0, 0);
    		V[1] = complexd(0, 0);
    		V[2] = complexd(0, 0);
   			V[3] = complexd(1, 0);
    	}
    	else if (size == 2){
    		if(rank == 0){
    			V[0] = complexd(0, 0);
    			V[1] = complexd(0, 0);
   		}
    		else{
    			V[0] = complexd(0, 0);
    			V[1] = complexd(1, 0);
   			}
    	} else{
    		if(rank != 3){
    			V[0] = complexd(0, 0);
    		} else{
   				V[0] = complexd(1, 0);
   			}
    	}
}


void canon_check(std::string name, int rank, complexd *V, MPI_Status &stat, int rc, int size) {
	if(size == 1){
    			cout << name << "RESULT " << V[0] << ' ' << V[1] << endl;
    }
	else {
		if(rank == 0){
			cout << name << "RESULT " << V[0] << ' ';
			MPI_Recv(&V[0], 1, MPI_COMPLEX, MPI_ANY_SOURCE, 8, MPI_COMM_WORLD, &stat);
			cout << V[0] << endl;
		}
		else{
			rc = MPI_Send(&V[0], 1, MPI_COMPLEX, 0, 8, MPI_COMM_WORLD);
		}    		
	}
}

void canon_check2(std::string name, int rank, complexd *V, MPI_Status &stat, int size, int rc) {
	if(size == 1){
    		cout << name << "RESULT " << V[0] << ' ' << V[1] << ' ' << V[2] << ' ' << V[3] << endl;
   		}
    	else if(size == 2){
    		if(rank == 0){
   				cout << name << "RESULT " << V[0] << ' ' << V[1] << ' ';
        		MPI_Recv(&V[0], 1, MPI_COMPLEX, MPI_ANY_SOURCE, 8, MPI_COMM_WORLD, &stat);
    			cout << V[0] << ' ';
    			MPI_Recv(&V[0], 1, MPI_COMPLEX, MPI_ANY_SOURCE, 8, MPI_COMM_WORLD, &stat);
    			cout << V[0] << endl;
   			}
   			else{
    			rc = MPI_Send(&V[0], 1, MPI_COMPLEX, 0, 8, MPI_COMM_WORLD);
    			rc = MPI_Send(&V[1], 1, MPI_COMPLEX, 0, 8, MPI_COMM_WORLD);
    		}
    	} else{
    		if(rank == 0){
    			cout << name << "RESULT " << V[0] << ' ';
       			MPI_Recv(&V[0], 1, MPI_COMPLEX, 1, 8, MPI_COMM_WORLD, &stat);
    			cout << V[0] << ' ';
   				MPI_Recv(&V[0], 1, MPI_COMPLEX, 2, 8, MPI_COMM_WORLD, &stat);
    			cout << V[0] << ' ';
    			MPI_Recv(&V[0], 1, MPI_COMPLEX, 3, 8, MPI_COMM_WORLD, &stat);
    			cout << V[0] << endl;
    		} else{
    			rc = MPI_Send(&V[0], 1, MPI_COMPLEX, 0, 8, MPI_COMM_WORLD);
    		}
    	}
}

int main(int argc, char **argv) {
    int was_read = 0;
    int test = 0;
    char *input, *output, *test_file;
    unsigned k, n, l;
    complexd *V;
    int box = 0;
    int canon = 0;
    complexd *need;
    complexd *need_new;
    for (int i = 1; i < argc; i++) { 
        string option(argv[i]);
        if(option.compare("Canonization") == 0){
        	canon = 1;
        	n = 2;
        }
        if(option.compare("Black_box") == 0){
        	box = 1;
        }
        if (option.compare("n") == 0) {
            n = atoi(argv[++i]);
        }
        if (option.compare("k") == 0) {
            k = atoi(argv[++i]);
        }
        if (option.compare("l") == 0) {
            l = atoi(argv[++i]);
        }
        if ((option.compare("file_read") == 0)) {
            input = argv[++i];
            was_read = 1;
        }
        if ((option.compare("file_write") == 0)) {
            output = argv[++i];
        }
        if ((option.compare("test") == 0)) {
            test = 1;
        }
        if ((option.compare("file_test") == 0)) {
            test_file = argv[++i];
        }
    }

    MPI_Init(&argc, &argv);
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    unsigned long long index = 1LLU << n;
    unsigned long long seg_size = index / size;
    need = (complexd*) malloc(sizeof(complexd) * seg_size);
    need_new = (complexd*) malloc(sizeof(complexd) * seg_size);
    MPI_Status stat;
    int rc;
    float fr_snd = 0;
    float ans = 0;
    if(box == 1){
    	V = generate_condition(seg_size, rank, size);

		std::string name = "NOT ";
    	NOT(k, V, rank, size, n);
		box_check(seg_size, V, name, rank, size, rc, stat);

		name = "ROT ";
        ROT(k, V, rank, size, n);
		box_check(seg_size, V, name, rank, size, rc, stat);

		name = "Hadamar ";
        Hadamar(k, V, rank, size, n);
		box_check(seg_size, V, name, rank, size, rc, stat);

		name = "CNOT ";
        CNOT(k, l, V, rank, size, n);
    	box_check(seg_size, V, name, rank, size, rc, stat);
		
		name = "CROT ";
        CROT(k, l, V, rank, size, n);
		box_check(seg_size, V, name, rank, size, rc, stat);

		name = "nHadamar ";
        nHadamar(k, l, V, rank, size, n);
    	box_check(seg_size, V, name, rank, size, rc, stat);


		if (rank == 0) {
			std::cout << std::endl << std::endl;
		}
        MPI_Finalize();
    	delete[] V;
    	return 0;
    }


    if(canon == 1){
    	complexd *V = new complexd[seg_size];
		std::string name = "NOT ";
		MPI_Status st;
    	if(size < 4){
    		if(rank == 0){
    			cout << "Вектор {0,0};{1, 0}" << endl;
    		}

			name = "NOT ";
    		my_vec_1(V, size, rank);
    		NOT(1, V, rank, size, 1);
			canon_check(name, rank, V, st, rc, size);

			name = "ROT ";
    		my_vec_1(V, size, rank);
    		ROT(1, V, rank, size, 1);
			canon_check(name, rank, V, st, rc, size);
    		
			name = "Hadamar ";
    		my_vec_1(V, size, rank);
    		Hadamar(1, V, rank, size, 1);
    		canon_check(name, rank, V, st, rc, size);
    	}

    	if(rank == 0){
    		cout << "Вектор {0,0};{0,0};{0,0};{1,0}" << endl;
    	}

		name = "CNOT ";
    	my_vec_2(V, size, rank);
   		CNOT(1, 2, V, rank, size, 2);
    	canon_check2(name, rank, V, st, size, rc);


		name = "CROT ";
    	my_vec_2(V, size, rank);
   		CROT(1, 2, V, rank, size, 2);
    	canon_check2(name, rank, V, st, size, rc);


    	name = "nHadamar ";
		my_vec_2(V, size, rank);
   		nHadamar(1, 2, V, rank, size, 2);
    	canon_check2(name, rank, V, st, size, rc);

		if (rank == 0) {
			std::cout << std::endl << std::endl;
		}
    	MPI_Finalize();
    	delete[] V;
    	return 0;
    }
    
    MPI_Finalize();
    delete[] V;
    delete[] need;
    delete[] need_new;
}