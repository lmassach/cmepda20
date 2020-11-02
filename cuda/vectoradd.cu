// Scriviamo in C, non in C++
// Compiliamo con
//     $ nvcc -o vectoradd vectoradd.cu --gpu-architecture=sm_35
// Sotto Windows serve 'x64 Native Tools Command Prompt for VS 2019'
//     $ nvcc -o vectoradd.exe vectoradd.cu --gpu-architecture=sm_35

#include <cuda.h>
#include <stdio.h>

#define N (1 << 20)

void random_vector(int *a, int nn) {
    for (int i = 0; i < nn; i++)
        a[i] = rand() % 100 + 1;
}

void vector_add_serial(int *a, int *b, int *res, int nn) {
    for (int i = 0; i < nn; i++)
        res[i] = a[i] + b[i];
}

// I kernel sono definiti __global__, le funzioni GPU chiamate dai kernel (non
// dall'host) sono definite __device__
__global__ void vector_add_gpu(int *a, int *b, int *res, int nn) {
    // Struttura dell'esecuzione: un elemento per blocco
    res[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

__global__ void vector_add_gpu_th(int *a, int *b, int *res, int nn) {
    // Struttura dell'esecuzione: un elemento per thread
    res[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

__global__ void vector_add_gpu_correct(int *a, int *b, int *res, int nn) {
    // Struttura dell'esecuzione: quella giusta
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    res[idx] = a[idx] + b[idx];
}

int main(void) {
    int *h_a, *h_b, *h_res; // Memoria allocata nell'host (h)
    int *d_a, *d_b, *d_res; // Memoria allocata nel device (d)
    int *h_res_check;
    int size = N * sizeof(int);
    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    h_a = (int*)malloc(size);
    h_b = (int*)malloc(size);
    h_res = (int*)malloc(size);

    random_vector(h_a, N);
    random_vector(h_b, N);

    // Sequenziale
    cudaEventRecord(start);
    cudaEventSynchronize(start);
    vector_add_serial(h_a, h_b, h_res, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Serial: %3.5f ms\n", time);
    for (int i = 0; i < 10; i++)
        printf("    %d + %d = %d\n", h_a[i], h_b[i], h_res[i]);
    printf("\n");

    // GPU
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_res, size);
    h_res_check = (int*)malloc(size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    // Questo non funziona: troppi blocchi!
    vector_add_gpu<<<N,1>>>(d_a, d_b, d_res, N);
    cudaDeviceSynchronize();
    cudaError_t kernel_error = cudaGetLastError();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("GPU: %3.5f ms\n", time);
    printf("Errors: %s\n", cudaGetErrorString(kernel_error));

    cudaMemcpy(h_res_check, d_res, size, cudaMemcpyDeviceToHost);
    int errors = 0;
    for (int i = 0; i < N; i++)
        if (h_res[i] != h_res_check[i])
            errors++;
    printf("%d errors/differences found\n", errors);
    for (int i = 0; i < 10; i++)
        printf("    %d + %d = %d\n", h_a[i], h_b[i], h_res_check[i]);
    printf("\n");

    // GPU (threads)
    cudaEventRecord(start);
    // Questo non funziona: troppi thread per blocco!
    vector_add_gpu_th<<<1,N>>>(d_a, d_b, d_res, N);
    kernel_error = cudaGetLastError();
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("GPU (threads): %3.5f ms\n", time);
    printf("Errors: %s\n", cudaGetErrorString(kernel_error));

    cudaMemcpy(h_res_check, d_res, size, cudaMemcpyDeviceToHost);
    errors = 0;
    for (int i = 0; i < N; i++)
        if (h_res[i] != h_res_check[i])
            errors++;
    printf("%d errors/differences found\n", errors);
    for (int i = 0; i < 10; i++)
        printf("    %d + %d = %d\n", h_a[i], h_b[i], h_res_check[i]);
    printf("\n");

    // GPU (correct)
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int nthreads = prop.maxThreadsPerBlock;
    int nblocks = N / nthreads; // Questo funziona perché N è 2^qualcosa!!!

    cudaEventRecord(start);
    // Questo non funziona: troppi thread per blocco!
    vector_add_gpu_correct<<<nblocks,nthreads>>>(d_a, d_b, d_res, N);
    kernel_error = cudaGetLastError();
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("GPU (correct): %3.5f ms\n", time);
    printf("Ran with %d threads, %d blocks\n", nthreads, nblocks);
    printf("Errors: %s\n", cudaGetErrorString(kernel_error));

    cudaMemcpy(h_res_check, d_res, size, cudaMemcpyDeviceToHost);
    errors = 0;
    for (int i = 0; i < N; i++)
        if (h_res[i] != h_res_check[i])
            errors++;
    printf("%d errors/differences found\n", errors);
    for (int i = 0; i < 10; i++)
        printf("    %d + %d = %d\n", h_a[i], h_b[i], h_res_check[i]);
    printf("\n");

    free(h_a);
    free(h_b);
    free(h_res);
    free(h_res_check);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);
    return 0;
}
