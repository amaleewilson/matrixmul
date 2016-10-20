
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>

#include <cublas_v2.h>
#include <cblas.h>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
  if (abort) exit(code);
  }
}


typedef struct MulData {
  int n, m, k;	// dimensions (indexed by i, j, k)
  float *a, *b;  // input arrays 
  float *c_from_gpu;  // results from the GPU
  float *c_from_cpu;  // results from the CPU
  float *c_from_openblas; // results from openBlas
  float *c_from_cublas; // resulst from cuBlas

  float *a_gpu, *b_gpu, *c_gpu;  // arrays in GPU memory
} MulData;


void init(MulData *d) {
  d->n = d->m = d->k = 1024;

  /* allocate memory for this baby */
  d->a = (float *) malloc(d->n * d->m * sizeof(float));
  d->b = (float *) malloc(d->m * d->k * sizeof(float));

  d->c_from_gpu = (float *) malloc(d->n * d->k * sizeof(float));
  d->c_from_cpu = (float *) malloc(d->n * d->k * sizeof(float));
  d->c_from_openblas = (float *) malloc(d->n * d->k * sizeof(float));
  d->c_from_cublas = (float *) malloc(d->n * d->k * sizeof(float));

  /* generate random data */
  srand((unsigned int)time(NULL));

  for (int j = 0; j < d->m; j++) {
    for (int i = 0; i < d->n; i++)
      d->a[j * d->m + i] = (float)rand() / (float)(RAND_MAX);
 
  for (int k = 0; k < d->k; k++)
    d->b[j * d->k + k] = (float)rand() / (float)(RAND_MAX);
  }
}


__global__ void matrix_mul(float *a_gpu, float *b_gpu, float *c_gpu,
		           int n, int m, int k){

  int c_i = blockIdx.x * blockDim.x + threadIdx.x;
  int c_k = blockIdx.y * blockDim.y + threadIdx.y;

  if (c_i >= n) return;
  if (c_k >= k) return;

  float mul_buffer = 0.0;
  for (int j = 0; j < m; j++)
    mul_buffer += a_gpu[j * n + c_i] * b_gpu[c_k * m + j];

  c_gpu[c_k * n + c_i] = mul_buffer;
}


void copy_to_gpu(MulData *d) {
  gpuErrchk( cudaMalloc( (void**) &d->a_gpu,
                         d->n * d->m * sizeof(float))
  );
  gpuErrchk( cudaMemcpy( d->a_gpu,
                         d->a,
                         d->n * d->m * sizeof(float),
                         cudaMemcpyHostToDevice)
  );

  gpuErrchk( cudaMalloc( (void**) &d->b_gpu,
                         d->m * d->k * sizeof(float))
  );

  gpuErrchk( cudaMemcpy( d->b_gpu,
                         d->b,
                         d->m * d->k * sizeof(float),
                         cudaMemcpyHostToDevice)
  );

  gpuErrchk( cudaMalloc( (void**) &d->c_gpu,
                         d->m * d->k * sizeof(float))
  );
}

void naive_gpu(MulData *d) {

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  dim3 dimBlock(16, 16);
  dim3 dimGrid( (int) ceil(d->n/dimBlock.y),
                (int) ceil(d->k/dimBlock.x)
  );

  matrix_mul<<<dimGrid, dimBlock>>>(d->a_gpu, d->b_gpu, d->c_gpu,
		                    d->n, d->m, d->k);

  gpuErrchk( cudaGetLastError() );
  cudaDeviceSynchronize();

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float miliseconds = 0.0;
  gpuErrchk( cudaMemcpy( d->c_from_gpu,
                         d->c_gpu,
                         d->k * d->n * sizeof(float),
                         cudaMemcpyDeviceToHost)
  );


  cudaEventElapsedTime(&miliseconds, start, stop);
  fprintf(stderr, "CUDA matrix mul took: %f [s]\n", miliseconds/1000);
}


void naive_cpu(MulData *d) {
  clock_t begin, end;
  double time_spent;

  begin = clock();
  int no_ops = 0;
  for (int i = 0; i < d->n; i++)
    for (int k = 0; k < d->k; k++) {

    float mul_buffer = 0;
    for (int j = 0; j < d->m; j++) {
      mul_buffer = mul_buffer + d->a[j * d->n + i] * d->b[k * d->m + j];
    }

    if (no_ops % 10000 == 0) {
      fprintf(stderr, "%d\n", no_ops);
      no_ops++;
    }

    d->c_from_cpu[k * d->n + i] = mul_buffer;
  }


  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

  fprintf(stderr, "CPU matrix mul took: %f [s]\n", time_spent);
}


void compare(float *res1, float *res2, int n, int k,
	     float precision, const char *comment) {
  fprintf(stderr, "compare %s\n", comment); 
  for (int i = 0; i < n * k; i++)
    if (abs(res1[i] - res2[i]) > precision)
      fprintf(stderr,
              "bad values at %d: %f != %f\n",
              i, res1[i], res2[i]);
  fprintf(stderr, "finished comapring %s\n", comment);
}


void mul_cblas(MulData *d) {
  struct timespec start, finish;
  double elapsed;

  clock_gettime(CLOCK_MONOTONIC, &start);

  cblas_sgemm(CblasColMajor,
              CblasNoTrans, CblasNoTrans,
              d->m, d->n, d->k, 1.0,
              d->a, d->m,
              d->b, d->k,
              0.0,
              d->c_from_openblas, d->m);

  clock_gettime(CLOCK_MONOTONIC, &finish);

  elapsed = (finish.tv_sec - start.tv_sec);
  elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

  fprintf(stderr, "BLAS matrix mul took: %f [s]\n", elapsed);
}


void mul_cublas(MulData *d) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  const float alpha = 1.0;
  const float beta = 0.0;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
              d->m, d->n, d->k, &alpha,
              d->a_gpu, d->m,
              d->b_gpu, d->k, &beta,
              d->c_gpu, d->m);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float miliseconds = 0.0;
  gpuErrchk( cudaMemcpy( d->c_from_cublas,
                         d->c_gpu,
                         d->k * d->n * sizeof(float),
                         cudaMemcpyDeviceToHost)
  );

  cudaEventElapsedTime(&miliseconds, start, stop);

  fprintf(stderr, "CUBLAS matrix mul took: %f [s]\n", miliseconds/1000);
}


int main() {
  cudaSetDevice(0);
//  MulData experiment;
//
//  init(&experiment);
//  naive_cpu(&experiment);
//
//  copy_to_gpu(&experiment);
//  naive_gpu(&experiment);
//
//  mul_cblas(&experiment);
//  mul_cublas(&experiment);

//  compare(experiment.c_from_openblas,
//          experiment.c_from_cublas,
//	  experiment.n, experiment.k,
//	  0.1,
//	  "openblas & cublas");

}
