#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void VectF1(double* IN, double* OUT, int n) {
    //printf("Thread %d starts executing F1\n", omp_get_thread_num());
    for (int i = 0; i < n; i++) {
        long int T = IN[i];
        OUT[i] = (double)(T % 4) + 0.5 + (IN[i] - trunc(IN[i]));
    }
    //printf("Thread %d ends executing F1\n", omp_get_thread_num());
}

void VectF2(double* IN, double* OUT, double v, int n) {
    //printf("Thread %d starts executing F2\n", omp_get_thread_num());
    for (int i = 0; i < n; i++) OUT[i] = v / (1.0 + fabs(IN[i]));
    //printf("Thread %d ends executing F2\n", omp_get_thread_num());
}

void VectScan(double* IN, double* OUT, int n) {
    double sum = 0.0;
    //printf("Thread %d starts executing Scan\n", omp_get_thread_num());
    for (int i = 0; i < n; i++) {
        sum += IN[i];
        OUT[i] = sum;  // Inclusive: include current element
    }
    //printf("Thread %d ends executing Scan\n", omp_get_thread_num());
}

void VectAverage(double* IN, double* OUT, int n) {
    //printf("Thread %d starts executing Average\n", omp_get_thread_num());
    for (int i = 1; i < n - 1; i++) {
        OUT[i] = (2.0 * IN[i] + IN[i - 1] + IN[i + 1]) / 4.0;
    }
    //printf("Thread %d ends executing Average\n", omp_get_thread_num());
}

double VectSum(double* V, int n) {
    double sum = 0;
    //printf("Thread %d starts executing Sum\n", omp_get_thread_num());
    for (int i = 0; i < n; i++) sum = sum + V[i];
    //printf("Thread %d ends executing Sum\n", omp_get_thread_num());
    return sum;
}

int main(int argc, char** argv) {
    int i, N = 20000, REP = 250000;

    // Get program arguments at runtime
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    if (argc > 2) {
        REP = atoi(argv[2]);
    }

    // Allocate memory space for arrays
    double* A = malloc(N * sizeof(double));
    double* B = malloc(N * sizeof(double));
    double* C = malloc(N * sizeof(double));
    double* D = malloc(N * sizeof(double));

    //  set initial values
    srand48(0);
    for (i = 0; i < N; i++) A[i] = drand48() - 0.5f;  // values between -0.5 and 0.5

    printf("Inputs: N= %d, Rep= %d\n", N, REP);

    double v = 10.0;
    for ( i=0; i<REP; i++ )
  { 
    #pragma omp parallel shared (A,B,C,D,v)
    {
      #pragma omp single
      VectF1 (A, B, N);
// -----------------------------
      #pragma omp sections  
      {
	      #pragma omp section
        VectF2      (B, C, v, N);
        #pragma omp section
        VectScan    (C, A, N);
      }
// -----------------------------
      VectAverage (B, D, N);
      #pragma omp barrier
      v = VectSum (D, N);
    }
  }

    printf("Outputs: v= %0.12e, A[%d]= %0.12e\n", v, N - 1, A[N - 1]);

    // Free memory space for arrays
    free(A);
    free(B);
    free(C);
    free(D);
}
