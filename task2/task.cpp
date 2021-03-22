#include <iostream>
#include <complex>
#include <cmath>
#include <random>
#include <fstream>
#include <mpi.h>

typedef std::complex<double> complexd;

bool arr_check(complexd *a1, complexd *a2, int size) {
    for (int i = 0; i < size; ++i) {
        if (a1[i] != a2[i]) {
            return false;
        }
    }
    return true;
}

complexd *read_from_file(const char *filename, unsigned &len, int &rank, int &comm_size) {
    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    int size = len / comm_size;
    complexd *arr = new complexd[size];
    double curr[2];
    int blocksize = 2 * size * rank * sizeof(double);
    MPI_File_seek(file, blocksize, MPI_SEEK_SET);
    for (int i = 0; i < size; ++i) {
        MPI_File_read(file, &curr, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
        arr[i].real(curr[0]);
        arr[i].imag(curr[1]);
    }
    MPI_File_close(&file);
    return arr;
}

void write_to_file(const char *filename, complexd *arr, unsigned &len, int &rank, int &comm_size) {
    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    int size = len / comm_size;
    double curr[2];
    int blocksize = 2 * size * rank * sizeof(double);
    MPI_File_seek(file, blocksize, MPI_SEEK_SET);
    for (int i = 0; i < size; ++i) {
        curr[0] = arr[i].real();
        curr[1] = arr[i].imag();
        MPI_File_write(file, &curr, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&file);
}

complexd *vec_gen(unsigned &len, int &rank, int &comm_size) {
    int size = len / comm_size;
    complexd *arr = new complexd[size];
    
    double sum = 0., whole_sum = 0.;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distr(0.0, 1.0);

    for (int i = 0; i < size; ++i) {
        arr[i] = complexd(distr(gen), distr(gen));
        double curr_abs = abs(arr[i]);
        sum += curr_abs * curr_abs;
    }

    //collecting sum from all procs
    MPI_Reduce(&sum, &whole_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        whole_sum = sqrt(whole_sum);
    }
    //broadcasting
    MPI_Bcast(&whole_sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (int i = 0; i < size; ++i) {
        arr[i] /= whole_sum;
    }
    return arr;
}

//transformation
void transform(complexd *vec, complexd *recv, complexd u[2][2], unsigned &len, int &k, int &rank, int &comm_size) {
    //len == 2^n, k - qbit number to transform
    int size = len / comm_size;
    int first_ind = rank * size;
    int rank_change = first_ind ^ (1u << (k - 1));
    rank_change /= size;

    if (rank != rank_change) {
        MPI_Sendrecv(vec, size, MPI_DOUBLE_COMPLEX, rank_change, 0, recv, size, MPI_DOUBLE_COMPLEX, rank_change, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rank > rank_change) {
            for (int i = 0; i < size; ++i) {
                recv[i] = u[1][0] * recv[i] + u[1][1] * vec[i];
            }
        } else {
            for (int i = 0; i < size; ++i) {
                recv[i] = u[0][0] * vec[i] + u[0][1] * recv[i];
            }
        }
    } else {
        unsigned shift = (int) log2(size) - k;
        unsigned pow = 1u << shift;
        for (int i = 0; i < size; ++i) {
            unsigned i0 = i & ~pow;
            unsigned i1 = i | pow;
            unsigned iq = (i & pow) >> shift;
            recv[i] = u[iq][0] * vec[i0] + u[iq][1] * vec[i1];
        }
    }
}

void write_time(int n, double time, int size, int k) {
    std::ofstream output("time.txt", std::ios_base::app);
    output << n << " " << " " << k << " " << size << " " << time << std::endl;
    output.close();
}


//n k test_flag(0 or 1) mode:0 - read from file, 1 - vector generation file_name(if mode == 0)
int main(int argc, char **argv) {
    int n, k, test, mode;
    sscanf(argv[1], "%d", &n);
    sscanf(argv[2], "%d", &k);
    sscanf(argv[3], "%d", &test);
    sscanf(argv[4], "%d", &mode);
    unsigned len = 1U << n; //2^n
    //преобразование Адамара
    //M_SQRT1_2 = 1 / sqrt(2);
    complexd u[2][2];
    u[0][0] = complexd(M_SQRT1_2);
    u[0][1] = complexd(M_SQRT1_2);
    u[1][0] = complexd(M_SQRT1_2);
    u[1][1] = complexd(-M_SQRT1_2);

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    complexd *vec;
    if (mode == 0) {
        vec = read_from_file(argv[5], len, rank, size);
    } else {
        vec = vec_gen(len, rank, size);
    }
    
    complexd *recv = new complexd[len / size];
    double start, end, time;
    start = MPI_Wtime();
    transform(vec, recv, u, len, k, rank, size);
    end = MPI_Wtime();
    time = end - start;

    if (test) {
        complexd *cmp_vec = read_from_file("test", len, rank, size);
        if (arr_check(vec, cmp_vec, len / size)) {
            std::cout << "Correct rank " << rank << std::endl;
        } else {
            std::cout << "Not correct rank " << rank << std::endl;
        }
    } else {
        write_to_file("test", recv, len, rank, size);
    }
    double total_time;
    MPI_Reduce(&time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        write_time(n, total_time, size, k);
    }
    delete[] vec;
    delete[] recv;
    MPI_Finalize();
    return 0;
}
