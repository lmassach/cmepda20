// Non è vero che scriviamo in C, è C++ (vedi la dichiarazione dei dim3...)
// Compiliamo con
//     $ nvcc -o matrixmul.out matrixmul.cu --gpu-architecture=sm_50
// Sotto Windows serve 'x64 Native Tools Command Prompt for VS 2019'
//     $ nvcc -o matrixmul.exe matrixmul.cu --gpu-architecture=sm_50
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define W_DEFAULT 1000ULL

// Macro per il controllo degli errori
inline void __malloc_assert(void* ptr, const char *file, int line) {
    if (ptr == NULL) {
        printf("On line %d of file %s: ", line, file);
        perror("malloc");
        exit(1);
    }
}
#define malloc_assert(ptr) { __malloc_assert(ptr, __FILE__, __LINE__); }

inline void __cuda_assert(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("On line %d of file %s: cuda: %s",
               line, file, cudaGetErrorString(err));
        exit(1);
    }
}
#define cuda_assert(err) { __cuda_assert(err, __FILE__, __LINE__); }

// Utilità
//inline size_t min(size_t a, size_t b) { return (a < b) ? a : b; }

// Moltiplica due matrici quadrate `m1` e `m2` larghe `width` e mette il
// risultato in `res`.
__global__ void ker_matrix_mul(float *m1, float *m2, float *res, size_t width) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < width && col < width) {
        res[row*width+col] = 0.0f;
        for (size_t k = 0; k < width; k++)
            res[row*width+col] += m1[row*width+k] * m2[k*width+col];
    }
}

// Come sopra, ma lavora serialmente con la CPU
void matrix_mul(float *m1, float *m2, float *res, size_t width) {
    for (size_t row = 0; row < width; row++) {
        for (size_t col = 0; col < width; col++) {
            res[row*width+col] = 0.0f;
            for (size_t k = 0; k < width; k++)
                res[row*width+col] += m1[row*width+k] * m2[k*width+col];
        }
    }
}

// Riempie `a` con numeri casuali in [0,1).
void random_vector(float *a, size_t n) {
    for (size_t i = 0; i < n; i++)
        a[i] = (rand() % 10000) / 10000.0f;
}

int main(int argc, char *argv[]) {
    size_t W = (argc > 1) ? strtoull(argv[1], NULL, 0) : W_DEFAULT;
    printf("Generating random %llu x %llu matrices...\n", W, W);

    // Dati e allocazione memoria
    float *h_m1, *h_m2, *h_res_cpu, *h_res_gpu;
    float *d_m1, *d_m2, *d_res;
    size_t size = W * W * sizeof(float);
    h_m1 = (float*)malloc(size); malloc_assert(h_m1);
    h_m2 = (float*)malloc(size); malloc_assert(h_m2);
    h_res_cpu = (float*)malloc(size); malloc_assert(h_res_cpu);
    h_res_gpu = (float*)malloc(size); malloc_assert(h_res_gpu);
    cuda_assert(cudaMalloc((void**)&d_m1, size));
    cuda_assert(cudaMalloc((void**)&d_m2, size));
    cuda_assert(cudaMalloc((void**)&d_res, size));

    // Misura del tempo
    float time, timecpy;
    cudaEvent_t start, stop, startcpy, stopcpy;
    cuda_assert(cudaEventCreate(&start));
    cuda_assert(cudaEventCreate(&stop));
    cuda_assert(cudaEventCreate(&startcpy)); // Con questi misuriamo anche il
    cuda_assert(cudaEventCreate(&stopcpy));  // tempo di copia della memoria

    // Inizializzazione dati (casuale)
    random_vector(h_m1, W * W);
    random_vector(h_m2, W * W);

    // Versione CPU/seriale
    cuda_assert(cudaEventRecord(start));
    cuda_assert(cudaEventSynchronize(start));
    matrix_mul(h_m1, h_m2, h_res_cpu, W);
    cuda_assert(cudaEventRecord(stop));
    cuda_assert(cudaEventSynchronize(stop));
    cuda_assert(cudaEventElapsedTime(&time, start, stop));
    printf("\nSerial: %3.5f ms\n", time);

    // Versione GPU/parallela
    cudaDeviceProp prop;
    cuda_assert(cudaGetDeviceProperties(&prop, 0));
    size_t maxth = prop.maxThreadsPerBlock;
    int nthx = min((size_t)prop.maxThreadsDim[0], W);
    int nthy = min(maxth / nthx, W);
    dim3 dim_block(nthx, nthy);
    // Queste sono divisioni intere (unsigned) arrotondate per eccesso
    dim3 dim_grid((W + nthx - 1) / nthx, (W + nthy - 1) / nthy);
    printf("\nGPU: max threads = %d (%d, %d, %d), blocks = (%d, %d, %d)\n",
           prop.maxThreadsPerBlock, prop.maxThreadsDim[0],
           prop.maxThreadsDim[1], prop.maxThreadsDim[2],
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Running with (%d, %d) blocks, (%d, %d) threads\n",
           dim_grid.x, dim_grid.y, dim_block.x, dim_block.y);

    cuda_assert(cudaEventRecord(startcpy));
    cuda_assert(cudaEventSynchronize(startcpy));
    cuda_assert(cudaMemcpy(d_m1, h_m1, size, cudaMemcpyHostToDevice));
    cuda_assert(cudaMemcpy(d_m2, h_m2, size, cudaMemcpyHostToDevice));

    cuda_assert(cudaEventRecord(start));
    ker_matrix_mul<<<dim_grid,dim_block>>>(d_m1, d_m2, d_res, W);
    cuda_assert(cudaGetLastError());
    cuda_assert(cudaDeviceSynchronize());
    cuda_assert(cudaEventRecord(stop));
    cuda_assert(cudaEventSynchronize(stop));

    cuda_assert(cudaMemcpy(h_res_gpu, d_res, size, cudaMemcpyDeviceToHost));
    cuda_assert(cudaEventRecord(stopcpy));
    cuda_assert(cudaEventSynchronize(stopcpy));

    cuda_assert(cudaEventElapsedTime(&time, start, stop));
    cuda_assert(cudaEventElapsedTime(&timecpy, startcpy, stopcpy));
    printf("GPU: %3.5f ms (%3.5f ms with memcpy)\n", time, timecpy);

    size_t diffs = 0, errs = 0;
    for (size_t i = 0; i < W * W; i++) {
        if (h_res_cpu[i] != h_res_gpu[i])
            diffs++;
        if (fabs(h_res_cpu[i] / h_res_gpu[i] - 1.0f) > 1e-5f)
            errs++;
    }
    printf("%llu differences found (%llu in excess of epsilon)\n", diffs, errs);

    // Cleanup
    free(h_m1); free(h_m2); free(h_res_cpu); free(h_res_gpu);
    cudaFree(d_m1); cudaFree(d_m2); cudaFree(d_res);
    return 0;
}
