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

// Usamos __restrict__ para indicar que los punteros no se solapan (ayuda al compilador)
__global__ void dev_laplace_error_opt(const float* __restrict__ A, 
                                      float* __restrict__ Anew, 
                                      float* __restrict__ error, 
                                      int n, int m) {
  // CORRECCIÓN DE COALESCENCIA:
  // Mapeamos .x a las columnas (j) porque es la dimensión contigua en memoria.
  // Mapeamos .y a las filas (i).
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int i = blockIdx.y * blockDim.y + threadIdx.y + 1;

  float local_diff = 0.0f;

  // Verificamos límites
  if (i < n - 1 && j < m - 1) {
      int idx = i * m + j;
      
      // Stencil computation
      float val_new = 0.25f * (A[idx - m] + A[idx + m] + A[idx - 1] + A[idx + 1]);
      Anew[idx] = val_new;
      
      // Calculamos error local
      local_diff = fabs(val_new - A[idx]);
  }

  // --- REDUCCIÓN OPTIMIZADA DEL ERROR ---
  
  // 1. Warp Reduction (Shuffle):
  // Cada hilo comparte su valor con otros hilos en el mismo "warp" (32 hilos)
  // sin usar memoria compartida, todo en registros.
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      float other_val = __shfl_down_sync(0xFFFFFFFF, local_diff, offset);
      local_diff = fmaxf(local_diff, other_val);
  }

  // 2. Block Reduction (Shared Memory):
  // El primer hilo de cada warp escribe su resultado parcial en memoria compartida.
  static __shared__ float s_max[32]; // Maximo 1024 hilos / 32 warps = 32 entradas
  int lane = threadIdx.x % warpSize;
  int warpId = threadIdx.x / warpSize; // Asumiendo blockDim 1D o linearizado

  // NOTA: Como usamos dim3(16, 16), threadIdx es 2D. Linealizamos para reducción:
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  lane = tid % 32;
  warpId = tid / 32;

  if (lane == 0) {
      s_max[warpId] = local_diff;
  }
  __syncthreads();

  // 3. El primer warp reduce los resultados de los otros warps
  if (warpId == 0) {
      // Leemos de shared mem (si el bloque es pequeño, rellenamos con 0)
      local_diff = (tid < (blockDim.x * blockDim.y) / 32) ? s_max[lane] : 0.0f;
      
      // Reducimos el último warp
      for (int offset = warpSize / 2; offset > 0; offset /= 2) {
          float other_val = __shfl_down_sync(0xFFFFFFFF, local_diff, offset);
          local_diff = fmaxf(local_diff, other_val);
      }
      
      // 4. Actualización Global Atómica (SOLO 1 vez por bloque)
      if (lane == 0) {
          atomicMax(error, local_diff);
      }
  }
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

  int iter = 0;
  int check_frequency = 1000; // Revisar error cada 100 iteraciones para reducir latencia PCIe

  while (error > tol && iter < iter_max) {
      // Reseteamos el error en GPU solo cuando vamos a chequear
      if (iter % check_frequency == 0) {
            cudaMemsetAsync(error_dev, 0, sizeof(float));
      }

      // Lanzamos el kernel
      dev_laplace_error_opt<<<gridDim, blockDim>>>(A_dev, Anew_dev, error_dev, n, m);

      // Swap de punteros (es solo intercambio de direcciones, muy rápido)
      float *swap = A_dev;
      A_dev = Anew_dev;
      Anew_dev = swap;

      // Solo copiamos el error de vuelta periódicamente
      if (iter % check_frequency == (check_frequency - 1)) {
          cudaMemcpy(&error, error_dev, sizeof(float), cudaMemcpyDeviceToHost);
          printf("Iter: %5d, Error actual: %0.6f\n", iter + 1, error);
          
          // Si el error ya es bajo, el while terminará en la siguiente evaluación
      }
      
      iter++;
  }

  cudaMemcpy(A, A_dev, n * m * sizeof(float), cudaMemcpyDeviceToHost);

  printf("Total Iterations: %5d, ERROR: %0.6f, ", iter, error);
  printf("A[%d][%d]= %0.6f\n", n / 128, m / 128, A[(n / 128) * m + m / 128]);

  cudaFree(A_dev);
  cudaFree(Anew_dev);
  cudaFree(error_dev);
  free(A);
}
