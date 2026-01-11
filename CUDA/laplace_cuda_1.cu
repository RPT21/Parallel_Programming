#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Custom version of atomicMax for float, since Nvidia does not support an
// official "atomicMax" function for floats
static inline __device__ float atomicMax(float *addr, float val) {
  unsigned int old = __float_as_uint(*addr), assumed;
  do {
    assumed = old;
    if (__uint_as_float(old) >= val)
      break;

    old = atomicCAS((unsigned int *)addr, assumed, __float_as_uint(val));
  } while (assumed != old);

  return __uint_as_float(old);
}

__global__ void dev_laplace_error(float* A, float* Anew, float* error, int n, int m) {
  // Set indices
  // Set to + 1 because computation starts at index 1
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;

  extern __shared__ float sdata[];

  if (i >= n - 1 || j >= m - 1) {
    return;
  }

  int idx = i*m + j;
  Anew[idx] = 0.25f*(A[idx-m]+A[idx+m]+A[idx-1]+A[idx+1]);

  float local_error = fabs(Anew[idx]-A[idx]);

  // Reduction in shared memory
  sdata[tid] = local_error;
  __syncthreads();

  // Perform reduction to find maximum error in the block
  for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
      if (tid < s)
          sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
      __syncthreads();
  }

  // Only one atomic operation per block
  if (tid == 0)
      atomicMax(error, sdata[0]);
  }

void laplace_init(float *in, int n, int m) {
  int i, j;
  const float pi = 2.0f * asinf(1.0f);
  memset(in, 0, n * m * sizeof(float));
  for (i = 0; i < m; i++)
    in[i] = 0.f;
  for (i = 0; i < m; i++)
    in[(n - 1) * m + i] = 0.f;
  for (j = 0; j < n; j++)
    in[j * m] = sinf(pi * j / (n - 1));
  for (j = 0; j < n; j++)
    in[j * m + m - 1] = sinf(pi * j / (n - 1)) * expf(-pi);
}

int main(int argc, char **argv) {
  int n = 4096, m = 4096;
  int iter_max = 10000, THREADS_BLOCK = 16;
  float *A;

  // const float tol = 1.0e-3f; (doing power of 2)
  const float tol = 1.0e-6f;
  float error = 1.0f;

  // get runtime arguments: n, m, iter_max and THREADS_BLOCK
  if (argc > 1) {
    n = atoi(argv[1]);
  }
  if (argc > 2) {
    m = atoi(argv[2]);
  }
  if (argc > 3) {
    iter_max = atoi(argv[3]);
  }
  if (argc > 4) {
    THREADS_BLOCK = atoi(argv[4]);
  }

  A = (float *)malloc(n * m * sizeof(float));

  //  set boundary conditions
  laplace_init(A, n, m);
  A[(n / 128) * m + m / 128] = 1.0f; // set singular point

  printf("Jacobi relaxation Calculation: %d x %d mesh,"
         " maximum of %d iterations. Threads per block= %d\n",
         n, m, iter_max, THREADS_BLOCK);

  float *A_dev, *Anew_dev; // Device pointers

  cudaMalloc(&A_dev, n * m * sizeof(float));
  cudaMalloc(&Anew_dev, n * m * sizeof(float));

  cudaMemcpy(A_dev, A, n * m * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Anew_dev, A, n * m * sizeof(float), cudaMemcpyHostToDevice);

  float *error_dev;
  cudaMalloc(&error_dev, sizeof(float));

  // Define grid and block dimensions
  // TODO: Define n and m for GPU run.
  int n_matrix_to_compute = n - 2;
  int m_matrix_to_compute = m - 2;
  dim3 gridDim((n_matrix_to_compute + THREADS_BLOCK - 1) / THREADS_BLOCK,
               (m_matrix_to_compute + THREADS_BLOCK - 1) / THREADS_BLOCK);
  dim3 blockDim(THREADS_BLOCK, THREADS_BLOCK);
  size_t shmem_size = THREADS_BLOCK * THREADS_BLOCK * sizeof(float);

  int iter = 0;
  while (error > tol && iter < iter_max) {
    iter++;

    cudaMemset(error_dev, 0, sizeof(float));
    // Fill parameters
    dev_laplace_error<<<gridDim, blockDim, shmem_size>>>(A_dev, Anew_dev, error_dev, n, m);

    cudaMemcpy(&error, error_dev, sizeof(float), cudaMemcpyDeviceToHost);

    // error = sqrtf(error);

    float *swap = A_dev;
    A_dev = Anew_dev;
    Anew_dev = swap; // swap pointers A_dev & Anew_dev

    if (iter % (iter_max / 10) == 0)
      printf("%5d, %0.6f\n", iter, error);
  }

  cudaMemcpy(A, A_dev, n * m * sizeof(float), cudaMemcpyDeviceToHost);

  printf("Total Iterations: %5d, ERROR: %0.6f, ", iter, error);
  printf("A[%d][%d]= %0.6f\n", n / 128, m / 128, A[(n / 128) * m + m / 128]);

  cudaFree(A_dev);
  cudaFree(Anew_dev);
  cudaFree(error_dev);
  free(A);
}
