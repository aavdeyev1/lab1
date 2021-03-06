#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// #include <time.h>
#include <pthread.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
 
typedef unsigned long long bignum;
__device__ int isPrimeGPU(bignum x);

// CUDA kernel. Each thread takes care of one element of c
__global__ void primeGPU(bignum *a, bignum *result, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n && id > 2 ) {
      if (id % 2 != 0)
        result[id] = isPrimeGPU(a[id]);
    }
}

__device__ int isPrimeGPU(bignum x) {

   bignum i;
   for (i = 2; i * i < x + 1; i++) {
       if (x % i == 0) {
           return 0;
       }
   }
   return 1;
}



void printArray(bignum * a, int len){
 
    int i;
    printf("\n[");
    for(i=0; i<len; i++){
    
       printf("%llu, ", a[i]);
 
    }
    printf("]\n");
 
 }
 
int main( int argc, char* argv[] )
{
    
    if(argc < 3)
    {
        printf("Usage: too few arguments\n");
        exit(-1);
    }
    // Retrieve N, blockSize from args
    bignum N = (bignum) (atoi(argv[1]) + 1);

   //  int odds = (int)ceil((double)(N + 1)/2);
    size_t bytes = (bignum)(N * sizeof(bignum));

    bignum *h_input;
    bignum *h_output;

    h_input = (bignum *)malloc(bytes);
    h_output = (bignum *)malloc(bytes);

    bignum *d_input;
    bignum *d_output;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

   //  int odds = ceil((double)(N + 1)/2)

    int i;
    for (i=0; i < N; i++){
      h_input[i] = i;
      h_output[i] = i; 
    }
    printArray(h_input, N);
    printArray(h_output, N);

    cudaMemcpy( d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;
 
    // Number of threads in each thread block
    blockSize = (int) atoi(argv[2]);
 
    // Number of thread blocks in grid
    gridSize = (int)ceil((double)((double)((N)/2)/blockSize));
 
    // Execute the kernel
    primeGPU<<<gridSize, blockSize>>>(d_input, d_output, N);
 
    // Copy array back to host
    cudaMemcpy( h_output, d_output, bytes, cudaMemcpyDeviceToHost );

    printArray(h_output, N);

    int total = 0;
    for (i=0; i < N; i++){
      total = total + h_output[i];
    }
    printf("Number of primes in that range: %d\n", total);
    
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);


   //  cudaDeviceReset();

    return 0;
}
