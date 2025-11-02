#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void VectF1(double* IN, double* OUT, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        long int T = IN[i];
        OUT[i] = (double)(T % 4) + 0.5 + (IN[i] - trunc(IN[i]));
    }
}

void VectF2(double* IN, double* OUT, double v, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) OUT[i] = v / (1.0 + fabs(IN[i]));
}

void VectScan(double* IN, double* OUT, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += IN[i];
        OUT[i] = sum;  // Inclusive: include current element
    }
}

/*
void VectScan(double* IN, double* OUT, int n) {
    #pragma omp parallel for scan(+: OUT[i])
    for (int i = 0; i < n; i++) {
        OUT[i] = IN[i];
    }
}
*/

void VectAverage(double* IN, double* OUT, int n) {
    #pragma omp parallel for
    for (int i = 1; i < n - 1; i++) {
        OUT[i] = (2.0 * IN[i] + IN[i - 1] + IN[i + 1]) / 4.0;
    }
}

/*
double VectSum(double* V, int n) {
    double sum = 0; 
    #pragma omp parallel shared(V, n)
{
    double sum_local = 0;
    #pragma omp for
    for (int i = 0; i < n; i++)
        sum_local += V[i];

    // #pragma omp atomic
    #pragma omp critical
    sum += sum_local;
}
    return sum;
}
*/

/*
double VectSum(double* V, int n) {
    double sum = 0.0;
    int i = 0;
    #pragma omp parallel for shared(V, n) private(i) reduction(+:sum) schedule(static,20000)
    for (i = 0; i < n; i++) {
        sum += V[i];
    }
    return sum;
}
*/

double VectSum(double* V, int n, int num_threads, double* partial_sums) {

    #pragma omp parallel shared(V, n, partial_sums, num_threads)
    {
        int tid = omp_get_thread_num();
        int chunk = (n + num_threads - 1) / num_threads; // ceil(n/num_threads)
        int start = tid * chunk;
        int end = (start + chunk > n) ? n : (start + chunk);

        double local_sum = 0;
        for (int i = start; i < end; i++)
            local_sum += V[i];

        // printf("Thread %d partial sum = %f\n", tid, local_sum);  // Debug, but we have repetitions REP too high

        partial_sums[tid] = local_sum;
    }

    // Suma secuencial de las sumas parciales
    double total = 0;
    for (int i = 0; i < num_threads; i++)
        total += partial_sums[i];

    return total;
}


int main(int argc, char** argv) {
    
    // omp_set_num_threads(6);
    // omp_set_dynamic(0);
    
    int i, N = 20000, REP = 250000;
    
    
    #pragma omp parallel
    {
        printf("Hilo %d de %d activos\n", omp_get_thread_num(), omp_get_num_threads());
    }
    

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

    int num_threads = omp_get_num_threads();
    double *partial_sums = (double*) malloc(num_threads * sizeof(double));

    //  set initial values
    srand48(0);
    for (i = 0; i < N; i++) A[i] = drand48() - 0.5f;  // values between -0.5 and 0.5

    printf("Inputs: N= %d, Rep= %d\n", N, REP);

    double v = 10.0;
    for (i = 0; i < REP; i++) {
        VectF1(A, B, N);
        VectF2(B, C, v, N);
        VectScan(C, A, N);
        VectAverage(B, D, N);
        v = VectSum(D, N, num_threads, partial_sums);
    }

    printf("Outputs: v= %0.12e, A[%d]= %0.12e\n", v, N - 1, A[N - 1]);

    // Free memory space for arrays
    free(A);
    free(B);
    free(C);
    free(D);
    free(partial_sums);
}
