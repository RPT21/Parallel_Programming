#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define PI (3.1415926535897932384626)

int main(int argc, char** argv){
  int n, m;
  if (argc > 1) n = atoi(argv[1]);
  if (argc > 2) m = atoi(argv[2]);
  int i, j, me, nproc, iter = 0;

  int iter_max = 100; //This is hardcoded
  const float tol = 1.0e-3f;  // Example tolerance (0.1%)
  float error = 1.0f;

  // The data is dynamically allocated
  float *A[n], *Anew[n];
  for (i = 0; i < n; i++){
    A[i] = (float*)malloc(m * sizeof(float));
    Anew[i] = (float*)malloc(m * sizeof(float));
  }

  // All the interior points in the 2D matrix are zero
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++) A[i][j] = 0;

  // set boundary conditions
  for (j = 0; j < m; j++){
    A[0][j] = 0.f;
    A[n - 1][j] = 0.f;
  }

  for (i = 0; i < n; i++){
    A[i][0] = sinf(PI * i / (n - 1));
    A[i][m - 1] = sinf(PI * i / (n - 1)) * expf(-PI);
  }

  /* Init. MPI */
  MPI_Status s;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  /* Setting each first and last row per process */
  int chunk = (n - 2 + nproc - 1) / nproc;
  int first_row = 1 + me * chunk;
  int last_row  = first_row + chunk - 1;
  if(last_row > n - 2) last_row = n - 2;

  //printf("I am %i. I live between %i and %i\n", me, first_row, last_row);
  //Communication between processes
  while(error > tol && iter < iter_max){
    if(me > 0){
        MPI_Sendrecv(A[first_row],   m, MPI_FLOAT, me-1, 66,
                     A[first_row-1], m, MPI_FLOAT, me-1, 67,
                     MPI_COMM_WORLD, &s);
    }
    if(me < nproc - 1){
        MPI_Sendrecv(A[last_row],    m, MPI_FLOAT, me+1, 67,
                     A[last_row+1],  m, MPI_FLOAT, me+1, 66,
                     MPI_COMM_WORLD, &s);
    }
    /* Computing functions from the Laplace algorithm */

    //Laplace step
    for(i = first_row; i <= last_row; i++){
      for(j = 1; j < m - 1; j++){
          Anew[i][j] = 
            (A[i][j + 1] + A[i][j - 1] + A[i - 1][j] + A[i + 1][j]) / 4;
      }
    }

    //Local error computation
    float error_loc = 0.0f;
    for (i = first_row; i <= last_row; i++){
      for (j = 1; j < m - 1; j++){
          error_loc = fmaxf(error_loc, sqrtf(fabsf(Anew[i][j] - A[i][j])));
      }
    }
    //Reducing all errors
    MPI_Allreduce(&error_loc, &error, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

    //Updating values
    for (i = first_row; i <= last_row; i++){
      for (j = 1; j < m - 1; j++){
          A[i][j] = Anew[i][j];
      } 
    }

    //Update index and print every 10 iterations
    iter++;
    if (iter % 10 == 0) printf("%5d, %0.6f. With love from %i, I live between %i and %i if you want to visit :) \n", iter, error, me, first_row, last_row);
  }

MPI_Finalize();
return 0;
}