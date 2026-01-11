/* This is an upgraded version from lap.c*/
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
// module load nvhpc/21.2
// nvc -fast -acc -tp=nehalem -gpu=cc80 -Minfo=accel lap.c -o LGPU
// sbatch -p cuda-ext.q -w aolin-gpu-3 --gres=gpu:1 -o ResGPU3.txt runGPU.txt LGPU

float stencil ( float v1, float v2, float v3, float v4 )
{
  return (v1 + v2 + v3 + v4) / 4.0f;
}

float laplace ( float *in, float *out, int n, int m )
{
    float error = 0.0f;
    #pragma acc parallel loop collapse(2) reduction(max:error) present(in[0:n*m], out[0:n*m])
    for ( int j = 1; j < n-1; j++ )
        for ( int i = 1; i < m-1; i++ ) {
        int idx = j*m + i;

        float newval = stencil(
            in[idx+1],
            in[idx-1],
            in[idx-m],
            in[idx+m]
        );
        out[idx] = newval;
        float diff = fabsf(newval - in[idx]);
        error = fmaxf(error, diff);
        }
  return error;
}

void laplace_init ( float *in, int n, int m )
{
  int i, j;
  const float pi = 2.0f * asinf(1.0f);
  memset(in, 0, n*m*sizeof(float));
  for (i=0; i<m; i++) {
    in[    i    ] = 0.f;
    in[(n-1)*m+i] = 0.f;
  }
  for (j=0; j<n; j++)  {
    in[   j*m   ] = sinf(pi*j / (n-1));
    in[ j*m+m-1 ] = in[j*m] * expf(-pi);
  }
}

int main(int argc, char** argv)
{
  int n = 4096, m = 4096;
  int iter_max = 10000;
  float *A, *Anew;
    
  const float tol = 1.0e-6f;
  float error= 1.0f;    

  // get runtime arguments: n, m and iter_max 
  if (argc>1) {  n        = atoi(argv[1]); }
  if (argc>2) {  m        = atoi(argv[2]); }
  if (argc>3) {  iter_max = atoi(argv[3]); }

  A    = (float*) malloc( n*m*sizeof(float) );
  Anew = (float*) malloc( n*m*sizeof(float) );

  //  set boundary conditions
  laplace_init (A, n, m);
  A[(n/128)*m+m/128] = 1.0f; // set singular point

  printf("Jacobi relaxation Calculation: %d x %d mesh,"
         " maximum of %d iterations\n", 
         n, m, iter_max );
  
  int iter = 0;
  float *cur = A; /*Pointers for switching */
  float *next = Anew;

  #pragma acc data copy(A[0:n*m]) create(Anew[0:n*m])
  {
    /* borders */
    #pragma acc parallel loop present(A[0:n*m], Anew[0:n*m])
    for (int i = 0; i < m; i++) {
        Anew[i] = A[i];
        Anew[(n-1)*m + i] = A[(n-1)*m + i];
    }

    #pragma acc parallel loop present(A[0:n*m], Anew[0:n*m])
    for (int j = 0; j < n; j++) {
        Anew[j*m] = A[j*m];
        Anew[j*m + (m-1)] = A[j*m + (m-1)];
    }

    while (error > tol && iter < iter_max)
    {
        iter++;

        error = laplace(cur, next, n, m);

        //swapping pointers
        float *tmp = cur; 
        cur = next; 
        next = tmp;

        if (iter % (iter_max/10) == 0) printf("%5d, %0.6f\n", iter, error);
    }
  }
  
  printf("Total Iterations: %5d, ERROR: %0.6f, ", sqrtf(error));
  printf("A[%d][%d]= %0.6f\n",
         n/128, m/128, A[(n/128)*m + m/128]);

  free(A); 
  free(Anew);
}

