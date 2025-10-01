// define n and m 
int n = 0;
int m = 0;
#define PI 3.14159265358979323846
#define A(i, j) A[(i)*(m) + (j)] // Posar parentesis provar sense i mirar que passa
#define Anew(i, j) Anew[(i)*(m) + (j)]

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
 
int main(int argc, char** argv) 
{ 

  if (argc != 3) {  // 1 (program) + 2 (arguments) = 3
        printf("Error: The correct use is -> %s <num1> <num2>\n", argv[0]);
        return 1; // Exit with error code
    }

  n = atoi(argv[1]);
  m = atoi(argv[2]);

  // declare global variables: A and Anew
  double *A = (double*)malloc(n * m * sizeof(double));
  double *Anew = (double*)malloc(n * m * sizeof(double));
  double *temp;


  // declare local variables: error, tol, iter_max ... 
  double error = 1.0;
  double tol = 1.0e-6;
  int iter_max = 1000;
  int iter = 0; 
  // get iter_max from command line at execution time 
  // set all values in matrix as zero  
  // set boundary conditions 

  for (int i=1; i<n-1; i++){
    for (int j=1; j<m-1; j++){
      A[i*m + j] = 0;
    }
  }

  for (int j=0; j<m; j++){
    A[j] = 0;
    A[(n-1)*m + j] = 0;

    Anew[j] = 0;
    Anew[(n-1)*m + j] = 0;
  }

  for (int i=0; i<n; i++){
    A[i*m] =sin(PI*i/(n-1));
    A[i*m + m-1] =sin(PI*i/(n-1))*exp(-PI);

    Anew[i*m] = A[i*m];
    Anew[i*m + m-1] = A[i*m + m-1];
  }

  // Copiar grain boundaries a la Anew i no haver de fer el memcpy
  
  // memcpy(Anew, A, n*m*sizeof(double));

  // Main loop: iterate until error <= tol a maximum of iter_max iterations

  while ( error > tol && iter < iter_max ) {
    error = 0; 
    for (int i=1; i<n-1; i++){
      for (int j=1; j<m-1; j++){
        // Canviar per el define que hi ha a dalt
        Anew[i*m + j] = 0.25 * (A[(i+1)*m + j] + A[(i-1)*m + j] + A[i*m + j+1] + A[i*m + j-1]);
        error = fmax(error, sqrt(fabs(Anew[i*m + j]-A[i*m + j])));
      }
    }
    //memcpy(A, Anew, n*m*sizeof(double));
    temp = A;
    A = Anew;
    Anew = temp;

    // Compute new values using main matrix and writing into auxiliary matrix 
    // Compute error = maximum of the square root of the absolute differences 
    // Copy from auxiliary matrix to main matrix 
    // if number of iterations is multiple of 10 then print error on the screen    
    iter++;
    if (iter % 10 == 0){
      printf("Iteration %d, error = %f\n", iter, error);
    }
  } // while 
  printf("Number of iterations: %d\n", iter);
  printf("Final error: %f\n", error);

  // print final matrix A
  for (int i=0; i<n; i++){
    for (int j=0; j<m; j++){
      printf("%f ", Anew[i*m + j]);
    }
    printf("\n");
  }

  // free A and Anew
  free(A);
  free(Anew);

  return 0;
}
