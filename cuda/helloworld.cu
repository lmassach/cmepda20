// Scriviamo in C, non in C++
// Compiliamo con
//     $ nvcc -o helloworld helloworld.cu --gpu-architecture=sm_35
// Sotto Windows serve 'x64 Native Tools Command Prompt for VS 2019'
//     $ nvcc -o helloworld.exe helloworld.cu --gpu-architecture=sm_35

#include <cuda.h>
#include <stdio.h>

__global__ void mykernel(void) {
    printf("Hello World! From GPU (block %d, thread %d)\n",
           blockIdx.x, threadIdx.x);
}

int main(void) {
    // Lancia con <<<N_BLOCCHI,N_THREADS>>>
    mykernel<<<1,5>>>();
    printf("Hello World! From host\n");
    cudaDeviceSynchronize();
    printf("Hello World! From host, after sync\n");
}
