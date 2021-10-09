#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// #include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
 
typedef unsigned long long bignum;

// CUDA kernel. Each thread takes care of one element of c
__global__ void vec1(bignum *a, bignum *result, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n)
        result[id] = a[id] + 1;
}


void printArray(bignum * a, int len){
 
    int i;
    printf("\n[`");
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
    bignum N = (bignum) atoi(argv[1]);
    int blockSize = (int) atoi(argv[2]);

    size_t bytes = N * sizeof(bignum);

    bignum *h_input;
    bignum *h_output;

    h_input = (bignum *)malloc(bytes);
    h_output = (bignum *)malloc(bytes);

    int i;
    for (i=0; i < N; i++){
      h_input[i] = i;
      h_output[i] = 0;    
    }
    printArray(h_input, N);
    printArray(h_output, N);

    
    free(h_input);
    free(h_output);



    return 0;
}
