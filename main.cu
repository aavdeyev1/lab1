#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
 
typedef unsigned long long bignum;

// CUDA kernel. Each thread takes care of one element of c
__global__ void vec1(double *a, double *result, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n)
        result[id] = a[id] + 1;
}


void printArray(char a[], int len){
 
    int i;
    printf("\n[");
    for(i=0; i<len; i++){
    
       printf("%d, ", a[i]);
 
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

    return 0;
}
