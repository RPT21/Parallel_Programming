#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define n 4096
#define m 4096

float A[n][m];
float Anew[n][m];
float y[n];

int main(int argc, char** argv)
{
    int i, j;
    int iter_max = 10000;
    
    const float pi  = 2.0f * asinf(1.0f);
    const float tol = 1.0e-8f;
    float error= 1.0f;    

    // get value of iter_max provided from command line at execution time
    if (argc>1) {  iter_max = atoi(argv[1]); }

    // set all values in matrix as zero
    memset(A, 0, n * m * sizeof(float));
    
    //  set boundary conditions: top and bottom rows
    for (i=0; i < m; i++)
    {
       A[0][i]   = 0.f;
       A[n-1][i] = 0.f;
    }

    //  set boundary conditions: left and right columns
    for (j=0; j < n; j++)
    {
       y[j] = sinf(pi * j / (n-1));
       A[j][0] = y[j];
       A[j][m-1] = y[j]*expf(-pi);
    }

    A[n/128][m/128] = 1.0f; // set singular point

    printf("Jacobi relaxation Calculation: %d x %d mesh, maximum of %d iterations\n", 
           n, m, iter_max );

    int iter = 0;

    #pragma acc data copy(A) create(Anew)
    while ( error > tol && iter < iter_max )
    {
       #pragma acc kernels
       for( i=1; i < m-1; i++ )
          for( j=1; j < n-1; j++)
              Anew[j][i] = ( A[j][i+1]+A[j][i-1]+A[j-1][i]+A[j+1][i]) / 4;

       error = 0.f;
       #pragma acc kernels
       for( i=1; i < m-1; i++ )
          for( j=1; j < n-1; j++)
              error = fmaxf( error, sqrtf( fabsf( Anew[j][i]-A[j][i] ) ) );

       #pragma acc kernels
       for( i=1; i < m-1; i++ )
          for( j=1; j < n-1; j++)
               A[j][i] = Anew[j][i];

       iter++;
       if( iter % (iter_max/10) == 0 ) printf("%5d, %0.6f\n", iter, error);
    }
    printf("Total Iterations: %5d, ERROR: %0.6f, ", iter, error);
    printf("A[%d][%d]= %0.6f\n", n/128, m/128, A[n/128][m/128]);

    return 0;
}
