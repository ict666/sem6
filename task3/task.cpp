#include <iostream>
#include <complex>
#include <cmath>
//#include <random>
#include <fstream>
#include <mpi.h>
#include <time.h>

#define EPS 0.01
typedef std::complex<double> complexd;

double normal_dis_gen()
{
    double S = 0.;
    for (int i = 0; i < 12; ++i) { 
        S += (double) rand() / RAND_MAX; 
    }
    return S - 6.0;
}

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
        complexd now = complexd(curr[0], curr[1]);
        arr[i] = now;
        //arr[i].real(curr[0]);
        //arr[i].imag(curr[1]);
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

/*complexd *vec_gen(unsigned &len, int &rank, int &comm_size) {
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
} */

complexd *vec_gen(unsigned &len, int &rank, int &comm_size) {
    srand(time(NULL));
    double abs = 0.0, abs_tot = 0.0;
    int size = len / comm_size;
    complexd *arr = new complexd[size];
    for (unsigned i = 0; i < size; i++) {
        arr[i] = complexd((std::rand() / (double) RAND_MAX - 0.5), (std::rand() / (double) RAND_MAX) - 0.5);
        abs += std::abs(arr[i]) * std::abs(arr[i]);
    }

    MPI_Reduce(&abs, &abs_tot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(&abs_tot, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    abs_tot = std::sqrt(abs_tot);
    for (unsigned i = 0; i < size; i++) {
        arr[i] /= abs_tot;
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

double n_transform(complexd *vec, complexd *recv, complexd u[2][2], unsigned &len, int &rank, int &comm_size, int n) {
    double start, end, time;
    start = MPI_Wtime();
    for (int q = 1; q < n + 1; ++q) {
        transform(vec, recv, u, len, q, rank, comm_size);
    }
    end = MPI_Wtime();
    time = end - start;
    return time;
}

double n_noise_transform(complexd *vec, complexd *recv, complexd u[2][2], unsigned &len, int &rank, int &comm_size, int n) {
    // q == qbit, p == thetta, v == u noised
    double p, start, end, time = 0;
    double start = MPI_Wtime();
    for (int q = 1; q < n + 1; ++q) {
        complexd v[2][2];
        if (rank == 0) {
            p = normal_dis_gen();
        }
        MPI_Bcast(&p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        p = p * EPS;
        v[0][0] = u[0][0] * cos(p) - u[0][1] * sin(p);
        v[0][1] = u[0][0] * sin(p) + u[0][1] * cos(p);
        v[1][0] = u[1][0] * cos(p) - u[1][1] * sin(p);
        v[1][1] = u[1][0] * sin(p) + u[1][1] * cos(p);
        start = MPI_Wtime();
        transform(vec, recv, v, len, q, rank, comm_size);
    }
    double end = MPI_Wtime();
    return end - start;
}

double scalar_product(complexd *ideal, complexd *noise, int rank, unsigned long long seg_size) {
    complexd sqr = 0;
    complexd sum = 0;
    for (unsigned long long i = 0; i < seg_size; ++i) {
        sqr += noise[i] * conj(ideal[i]);
    }
    MPI_Reduce(&sqr, &sum, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        return abs(sum) * abs(sum);
    } else {
        return 0;
    }
}

void write_time(int n, double time, int size) {
    std::ofstream output("time.txt", std::ios_base::app);
    output << n << " " << size << " " << time << std::endl;
    output.close();
}

void acc_to_file(double acc) {
    std::ofstream output("accuracy.txt", std::ios_base::app);
    output << acc << std::endl;
    output.close();
}

//n mode:0 - read from file, 1 - vector generation file_name(if mode == 0)
int main(int argc, char **argv) {
    int n, test, mode;
    sscanf(argv[1], "%d", &n);
    sscanf(argv[2], "%d", &mode);
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
    
    complexd *recv_ideal = new complexd[len / size];
    complexd *recv_noised = new complexd[len / size];
    //simple n transform
    double simple_transform_time = n_transform(vec, recv_ideal, u, len, rank, size, n);
    //noise n transform
    double noise_transform_time = n_noise_transform(vec, recv_noised, u, len, rank, size, n);


    double scalar_dist = scalar_product(recv_ideal, recv_noised, rank, len / size);
    if (rank == 0) {
        //write distance to file
        //1 - F == 1 - scalar_dist
        acc_to_file(1 - scalar_dist);
    }

    double total_time;
    MPI_Reduce(&noise_transform_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        write_time(n, total_time, size);
    }
    delete[] vec;
    delete[] recv_ideal;
    delete[] recv_noised;
    MPI_Finalize();
    return 0;
}
