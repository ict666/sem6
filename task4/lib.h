#include <mpi.h>
#include <iostream>
#include <complex>
using namespace std;

typedef complex<double> complexd;

void OneQubitEvolution(complexd *in, complexd *out, complexd U[2][2], int n, int q, int rank, unsigned long long seg_size) {
    int first_index = rank * seg_size;
    int rank_change = first_index ^(1u << (q - 1));     
    rank_change /= seg_size;
    int ss = seg_size;
    if (rank != rank_change) {
        int rc;
        MPI_Status stat3;
        if (rank < rank_change) {
            rc = MPI_Send(in, seg_size, MPI_DOUBLE_COMPLEX, rank_change, 10, MPI_COMM_WORLD);
            MPI_Recv(out, seg_size, MPI_DOUBLE_COMPLEX, rank_change, 10, MPI_COMM_WORLD, &stat3);
            for (int i = 0; i < seg_size; i++) {
                out[i] = U[0][0] * in[i] + U[0][1] * out[i];
            }
        } else {
            MPI_Recv(out, seg_size, MPI_DOUBLE_COMPLEX, rank_change, 10, MPI_COMM_WORLD, &stat3);
            rc = MPI_Send(in, seg_size, MPI_DOUBLE_COMPLEX, rank_change, 10, MPI_COMM_WORLD);
            for (int i = 0; i < seg_size; i++) {
                out[i] = U[1][0] * out[i] + U[1][1] * in[i];
            }
        }
    } else {
        int cr = 0;
        while(ss != 1){
            ss /= 2;
            cr++;
        }
        int shift = cr - q;
        int pow = 1 << (shift);
        for (int i = 0; i < seg_size; i++) {
            int i0 = i & ~pow;
            int i1 = i | pow;
            int iq = (i & pow) >> shift;
            out[i] = U[iq][0] * in[i0] + U[iq][1] * in[i1];
        }
    }
    for (int i = 0; i < seg_size; i++){
    	in[i] = out[i];
	}
}


void TwoQubitEvolution(complexd *buf0, complexd *buf1, complexd *buf2, complexd *buf3, complexd U[4][4], unsigned int n,
                       unsigned int k, unsigned int l, int rank, int seg_size) {
    unsigned N = 1u << n;
    complexd *buf_ans;
    buf_ans = new complexd[seg_size];
    unsigned first_index = rank * seg_size;
    if(l > k){
       	int buffer = k;
       	k = l;
       	l = buffer;
    }
    unsigned rank1_change = first_index ^(1u << (k - 1)); 
    unsigned rank2_change = first_index ^(1u << (l - 1));     
	unsigned rank3_change = first_index ^((1u << (k - 1)) | (1u << (l - 1))); 

    rank1_change /= seg_size;
    rank2_change /= seg_size;
    rank3_change /= seg_size;
    if (rank == rank1_change) { 
        int shift1=k - 1;
		int shift2=l -1;
		int q1=1<<(shift1);
		int q2=1<<(shift2);

        for (int i = rank * seg_size; i < rank*seg_size + seg_size; i++) {
        	int first = i & ~q1 & ~q2;
			int second = i & ~q1 | q2;
			int third = (i | q1) & ~q2;
			int fourth = i | q1 | q2;	
			int iq1 = (i & q1) >> shift1;
			int iq2 = (i & q2) >> shift2;
			int iq=(iq1<<1)+iq2;
		
			buf_ans[i - rank*seg_size] = U[iq][(0<<1)+0] * buf0[first - rank * seg_size] + U[iq][(0<<1)+1] * buf0[second - rank * seg_size] + U[iq][(1<<1)+0] * buf0[third - rank * seg_size] + U[iq][(1<<1)+1] * buf0[fourth - rank * seg_size];
        }
    } else if (rank == rank2_change) { 
        MPI_Sendrecv(buf0, seg_size, MPI_DOUBLE_COMPLEX, rank1_change, 0, buf1, seg_size, MPI_DOUBLE_COMPLEX,
                     rank1_change, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int shift1=k - 1;
		int shift2=l -1;
		int q1=1<<(shift1);
		int q2=1<<(shift2);
        for (int i = seg_size*rank; i < rank*seg_size + seg_size; i++) {


        	int first = i & ~q1 & ~q2;
			int second = i & ~q1 | q2;
			int third = (i | q1) & ~q2;
			int fourth = i | q1 | q2;
			int iq1 = (i & q1) >> shift1;
			int iq2 = (i & q2) >> shift2;

			int iq=(iq1<<1)+iq2;
			if(first < seg_size * rank || first >= seg_size * (rank + 1)){
				first -= rank1_change * seg_size;
				second -= rank1_change * seg_size;
				third -= rank * seg_size;
				fourth -= rank * seg_size;
				buf_ans[i - seg_size * rank] = U[iq][(0<<1)+0] * buf1[first] + U[iq][(0<<1)+1] * buf1[second] + U[iq][(1<<1)+0] * buf0[third] + U[iq][(1<<1)+1] * buf0[fourth];
			} else{
				third -= rank1_change * seg_size;
				fourth -= rank1_change * seg_size;
				first -= rank * seg_size;
				second -= rank * seg_size;
				//cout << i << endl;
				buf_ans[i - seg_size * rank] = U[iq][(0<<1)+0] * buf0[first] + U[iq][(0<<1)+1] * buf0[second] + U[iq][(1<<1)+0] * buf1[third] + U[iq][(1<<1)+1] * buf1[fourth];

			}
        }

    } else { 
        MPI_Sendrecv(buf0, seg_size, MPI_DOUBLE_COMPLEX, rank1_change, 0, buf1, seg_size, MPI_DOUBLE_COMPLEX, rank1_change, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(buf0, seg_size, MPI_DOUBLE_COMPLEX, rank2_change, 0, buf2, seg_size, MPI_DOUBLE_COMPLEX, rank2_change, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(buf0, seg_size, MPI_DOUBLE_COMPLEX, rank3_change, 0, buf3, seg_size, MPI_DOUBLE_COMPLEX, rank3_change, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int shift1=k - 1;
		int shift2=l -1;
		int q1=1<<(shift1);
		int q2=1<<(shift2);

        for (int i = seg_size * rank; i < seg_size * rank + seg_size; i++) {
           	int first = i & ~q1 & ~q2;
			int second = i & ~q1 | q2;
			int third = (i | q1) & ~q2;
			int fourth = i | q1 | q2;
			int iq1 = (i & q1) >> shift1;
			int iq2 = (i & q2) >> shift2;
			int iq=(iq1<<1)+iq2;
			buf_ans[i - seg_size * rank] = 0;
			if(rank*seg_size <= first && first < rank*seg_size + seg_size){
				first -= rank*seg_size;
				buf_ans[i - seg_size * rank] += U[iq][(0<<1)+0] * buf0[first];
			} else if (rank1_change *seg_size <= first && first < rank1_change *seg_size + seg_size){
				first -= rank1_change *seg_size;
				buf_ans[i - seg_size * rank] += U[iq][(0<<1)+0] * buf1[first];
            } else if (rank2_change *seg_size <= first && first < rank2_change *seg_size + seg_size){
            	first -= rank2_change *seg_size;
				buf_ans[i - seg_size * rank] += U[iq][(0<<1)+0] * buf2[first];
			} else if (rank3_change *seg_size <= first && first< rank3_change *seg_size + seg_size){
				first -= rank3_change *seg_size;
				buf_ans[i - seg_size * rank] += U[iq][(0<<1)+0] * buf3[first];
			}


			if(rank*seg_size <= second && second< rank*seg_size + seg_size){
				second -= rank*seg_size;
				buf_ans[i - seg_size * rank] += U[iq][(0<<1)+1] * buf0[second];
			} else if (rank1_change *seg_size <= second && second< rank1_change *seg_size + seg_size){
				second -= rank1_change*seg_size;
				buf_ans[i - seg_size * rank] += U[iq][(0<<1)+1] * buf1[second];
            } else if (rank2_change *seg_size <= second && second< rank2_change *seg_size + seg_size){
            	second -= rank2_change*seg_size;
				buf_ans[i - seg_size * rank] += U[iq][(0<<1)+1] * buf2[second];
			} else if (rank3_change *seg_size <= second && second< rank3_change *seg_size + seg_size){
				second -= rank3_change*seg_size;
				buf_ans[i - seg_size * rank] += U[iq][(0<<1)+1] * buf3[second];
			}


			if(rank*seg_size <= third && third< rank*seg_size + seg_size){
				third -= rank*seg_size;
				buf_ans[i - seg_size * rank] += U[iq][(1<<1)+0] * buf0[third];
			} else if (rank1_change *seg_size <= third && third< rank1_change *seg_size + seg_size){
				third -= rank1_change*seg_size;
				buf_ans[i - seg_size * rank] += U[iq][(1<<1)+0] * buf1[third];
            } else if (rank2_change *seg_size <= third && third< rank2_change *seg_size + seg_size){
            	third -= rank2_change*seg_size;
				buf_ans[i - seg_size * rank] += U[iq][(1<<1)+0] * buf2[third];
			} else if (rank3_change *seg_size <= third && third< rank3_change *seg_size + seg_size){
				third -= rank3_change*seg_size;
				buf_ans[i - seg_size * rank] += U[iq][(1<<1)+0] * buf3[third];
			}


			if(rank*seg_size <= fourth && fourth< rank*seg_size + seg_size){
				fourth -= rank*seg_size;
				buf_ans[i - seg_size * rank] += U[iq][(1<<1)+1] * buf0[fourth];
			} else if (rank1_change *seg_size <= fourth && fourth< rank1_change *seg_size + seg_size){
				fourth -= rank1_change*seg_size;
				buf_ans[i - seg_size * rank] += U[iq][(1<<1)+1] * buf1[fourth];
            } else if (rank2_change *seg_size <= fourth && fourth < rank2_change *seg_size + seg_size){
            	fourth -= rank2_change*seg_size;
				buf_ans[i - seg_size * rank] += U[iq][(1<<1)+1] * buf2[fourth];
			} else if (rank3_change *seg_size <= fourth && fourth< rank3_change *seg_size + seg_size){
				fourth -= rank3_change*seg_size;
				buf_ans[i - seg_size * rank] += U[iq][(1<<1)+1] * buf3[fourth];
			}



        }
    }
    for (int i = 0; i < seg_size; i++){
    	buf0[i] = buf_ans[i];
	}
}


void NOT(unsigned k, complexd *buf0, int rank, int size, unsigned n) {
    complexd *buf1;
    unsigned N = 1u << n;

    unsigned seg_size = N / size;
    buf1 = new complexd[seg_size];
    complexd U[2][2];
    U[0][0] = 0;
    U[0][1] = 1;
    U[1][0] = 1;
    U[1][1] = 0;
    OneQubitEvolution(buf0, buf1, U, n, k, rank, seg_size);

}


void CNOT(unsigned k, unsigned l, complexd *buf0, int rank, int size, unsigned n) {
    complexd *buf1, *buf2, *buf3;
    unsigned N = 1u << n;
    unsigned seg_size = N / size;
    buf1 = new complexd[seg_size];
    buf2 = new complexd[seg_size];
    buf3 = new complexd[seg_size];
    complexd U[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            U[i][j] = 0;
        }
    }
    U[0][0] = 1;
    U[1][1] = 1;
    U[2][3] = 1;
    U[3][2] = 1;

    
    TwoQubitEvolution(buf0, buf1, buf2, buf3, U, n, k, l, rank, seg_size);
}


void Hadamar(unsigned k, complexd *buf0, int rank, int size, unsigned n) {
    complexd *buf1;
    unsigned N = 1u << n;
    unsigned seg_size = N / size;
    buf1 = new complexd[seg_size];
    complexd U[2][2];
    U[0][0] = 1 / (sqrt(2));
    U[0][1] = 1 / (sqrt(2));
    U[1][0] = 1 / (sqrt(2));
    U[1][1] = -1 / (sqrt(2));
    OneQubitEvolution(buf0, buf1, U, n, k, rank, seg_size);
}

void nHadamar(unsigned k, unsigned l, complexd *buf0, int rank, int size, unsigned n) {
    complexd *buf1, *buf2, *buf3;
    unsigned N = 1u << n;
    unsigned seg_size = N / size;
    buf1 = new complexd[seg_size];
    buf2 = new complexd[seg_size];
    buf3 = new complexd[seg_size];
    complexd U[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            U[i][j] = 1. / 2;
        }
    }
    U[1][3] = -1. / 2;
    U[1][1] = -1. / 2;
    U[2][2] = -1. / 2;
    U[2][3] = -1. / 2;
    U[3][2] = -1. / 2;
    U[3][1] = -1. / 2;
    
    TwoQubitEvolution(buf0, buf1, buf2, buf3, U, n, k, l, rank, seg_size);
}

void ROT(unsigned k, complexd *buf0, int rank, int size, unsigned n) {
    complexd *buf1;
    unsigned N = 1u << n;
    unsigned seg_size = N / size;
    buf1 = new complexd[seg_size];
    complexd U[2][2];
    U[0][0] = 1;
    U[0][1] = 0;
    U[1][0] = 0;
    U[1][1] = -1;
    OneQubitEvolution(buf0, buf1, U, n, k, rank, seg_size);
}

void CROT(unsigned k, unsigned l, complexd *buf0, int rank, int size, unsigned n) {
    complexd *buf1, *buf2, *buf3;
    unsigned N = 1u << n;
    unsigned seg_size = N / size;
    buf1 = new complexd[seg_size];
    buf2 = new complexd[seg_size];
    buf3 = new complexd[seg_size];
    complexd U[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            U[i][j] = 0;
        }
    }
    U[0][0] = 1;
    U[1][1] = 1;
    U[2][2] = 1;
    U[3][3] = -1;
    TwoQubitEvolution(buf0, buf1, buf2, buf3, U, n, k, l, rank, seg_size);
}