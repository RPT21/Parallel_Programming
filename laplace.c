#define PI 3.14159265358979323846

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
 
int main(int argc, char** argv){
  if (argc != 5) {  
        printf("Error: The correct use is -> %s <n> <m> <iter_max> <tolerance>\n", argv[0]);
        return 1;
    }

  int n = atoi(argv[1]);
  int m = atoi(argv[2]);
  int iter_max = atoi(argv[3]);
  float tol = (float)atof(argv[4]);

  float *A = (float*)malloc(n * m * sizeof(double));
  float *Anew = (float*)malloc(n * m * sizeof(double));
  float *temp;

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
    A[i*m] =sinf(PI*i/(n-1));
    A[i*m + m-1] =sinf(PI*i/(n-1))*expf(-PI);

    Anew[i*m] = A[i*m];
    Anew[i*m + m-1] = A[i*m + m-1];
  }

  float error = 1.0;
  int iter = 0;
  tol = tol*tol;

  while ( error > tol && iter < iter_max ) {
    error = 0; 
    for (int i=1; i<n-1; i++){
      for (int j=1; j<m-1; j++){
        Anew[i*m + j] = 0.25 * (A[(i+1)*m + j] + A[(i-1)*m + j] + A[i*m + j+1] + A[i*m + j-1]);
        error = fmaxf(error, fabsf(Anew[i*m + j]-A[i*m + j]));
      }
    }

    temp = A;
    A = Anew;
    Anew = temp;

    iter++;

    if (iter % 10 == 0){
      printf("Iteration %d, error = %f\n", iter, error);
    }
  }

  printf("Number of iterations: %d\n", iter);
  printf("Final error: %f\n", error);

  free(A);
  free(Anew);

  return 0;
}
