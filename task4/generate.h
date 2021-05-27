#include <mpi.h>
#include <iostream>
#include <complex>

typedef complex<double> complexd;

complexd *read(char *f, int rank, unsigned long long seg_size) {
    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, f, MPI_MODE_RDONLY, MPI_INFO_NULL,&file);
    complexd *A;
    A = (complexd*) malloc(sizeof(complexd) * seg_size);
    double d[2];
    MPI_File_seek(file, 2 * seg_size * rank * sizeof(double), MPI_SEEK_SET);
    for (int i = 0; i < seg_size; ++i) {
        MPI_File_read(file, &d, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
        A[i] = complexd(d[0], d[1]);
    }
    MPI_File_close(&file);
    return A;
}

void write(char *f, complexd *B, int n, int rank, int size, unsigned long long seg_size) {
    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, f, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    double d[2];
    MPI_File_seek(file, 2 * seg_size * rank * sizeof(double), MPI_SEEK_SET);
    for (int i = 0; i < seg_size; ++i) {
        d[0] = B[i].real();
        d[1] = B[i].imag();
        MPI_File_write(file, &d, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&file);
}

complexd* generate_condition(unsigned long long seg_size, int rank, int size){ 
    double module = 0;
    unsigned int seed = time(NULL) + rank;
    complexd *V = new complexd[seg_size];
    for (long long unsigned  i = 0; i < seg_size; i++){
        V[i] = complexd(rand_r(&seed)%100 + 1.0, rand_r(&seed)%100 + 1.0);
        module += abs(V[i] * V[i]);
    }
    int rc;
    double new_m;
    MPI_Status stat;
    if(rank != 0){
        module += 1;
        rc = MPI_Send(&module, 1, MPI_DOUBLE, 0, 999, MPI_COMM_WORLD);
        MPI_Recv(&module, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 3, MPI_COMM_WORLD, &stat);
    }
    else{
        for(int i = 1; i < size; i++){
            MPI_Recv(&new_m, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 999, MPI_COMM_WORLD, &stat);
            module += new_m;
        }
        module = sqrt(module);
        for(int i = 1; i < size; i++){
            rc = MPI_Send(&module, 1, MPI_DOUBLE, i, 3, MPI_COMM_WORLD);
        }
    }
    for (long long unsigned j = 0; j < seg_size; j++) {
        V[j] /= module;
    }
    return V;
}