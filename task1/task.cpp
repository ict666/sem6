#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <complex>
#include <vector>
#include <fstream>
#include <omp.h>

typedef std::complex<double> complexd;

double drand(double min, double max) {
    double d = (double) rand() / RAND_MAX;
    return min + d * (max - min);
}

void vec_gen(complexd *vec, unsigned long long size, int &threads) {
    double sum = 0;

#pragma omp parallel for reduction(+: sum) num_threads(threads)
    for (unsigned long long i = 0; i < size; ++i) {
        vec[i] = complexd(drand(20, 30), drand(20, 30));
        double len = abs(vec[i]);
        sum += len * len;
    }

    sum = sqrt(sum);
    //нормировка
#pragma omp parallel for num_threads(threads)
    for (unsigned long long i = 0; i < size; ++i) {
        vec[i] /= sum;
    }    
}

void transform(complexd *vec, complexd *out, unsigned long long &size, int &n, int &k, complexd u[2][2], int &threads) {
    // n - число кубитов
    // k - номер кубита для преобразования
    unsigned long long shift = n - k;
    unsigned long long mask = 1 << shift;
#pragma omp parallel for schedule(static) num_threads(threads)
    for (unsigned long long i = 0; i < size; ++i) {
        unsigned long long i0 = i & ~mask;
        unsigned long long i1 = i | mask;
        unsigned long long iq = (i & mask) >> shift;
        out[i] = u[iq][0] * vec[i0] + u[iq][1] * vec[i1];
    }
}

void print_vec(complexd *vec, unsigned long long size) {
    for (unsigned long long i = 0; i < size; ++i) {
        std::cout << vec[i] << std::endl;
    }
}

void output(int &n, double &time, int &threads) {
    std::ofstream fout;
    fout.open("out.txt", std::ios_base::app);
    fout << n << " " << threads << " " << time << std::endl;
    fout.close();
}

//input: n - number of qbits, k - number of qbit to transform, U - complex matrix, v - complex vector (size = 2^n)
int main(int argc, char **argv) {
    srand(time(NULL));

    int n;
    int k;
    int threads;
    sscanf(argv[1], "%d", &n);
    sscanf(argv[2], "%d", &k);
    sscanf(argv[3], "%d", &threads);

    //преобразование Адамара
    //M_SQRT1_2 = 1 / sqrt(2);
    complexd u[2][2];
    u[0][0] = complexd(M_SQRT1_2);
    u[0][1] = complexd(M_SQRT1_2);
    u[1][0] = complexd(M_SQRT1_2);
    u[1][1] = complexd(-M_SQRT1_2);
    unsigned long long size = 1;
    //size = 2^n
    if (n > 0) {
        size = 1 << n;
    }
    complexd *vec = new complexd[size];
    complexd *out = new complexd[size];
    vec_gen(vec, size, threads);

    double start, end, time;
    start = omp_get_wtime();
    transform(vec, out, size, n, k, u, threads);
    end = omp_get_wtime();
    time = end - start;


    output(n, time, threads);

    /*std::cout << "INPUT: " << std::endl;
    print_vec(vec, size);
    std::cout << std::endl << "OUTPUT: " << std::endl;
    print_vec(out, size);
    std::cout << "TIME: " << time << std::endl; */

    delete[] vec;
    delete[] out;
    return 0;
}
