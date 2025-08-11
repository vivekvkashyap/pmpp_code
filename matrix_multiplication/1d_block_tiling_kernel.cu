#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 256
#define TILE_WIDTH 16
#define TM 4  

void matrix_mul_cpu(float *A, float *B, float *C, int n){
    for (int i=0; i<n; i++){
        for (int j=0; j<n; j++){
            float sum = 0.0f;
            for (int k=0; k<n; k++){
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

__global__ void block_tiling_1d_matrix_mul(float *A, float *B, float *C, int n) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const int BK = TM;
    const int BN = TILE_WIDTH;
    const int BM = TILE_WIDTH;

    // each warp will calculate 32*TM elements, with 32 being the columnar dim.
    const int threadCol = threadIdx.x % BN;
    const int threadRow = threadIdx.x / BN;

    const int K = n;
    // allocate space for the current blocktile in SMEM
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * n + cCol * BN;

    // todo: adjust this to each thread to load multiple entries and
    // better exploit the cache sizes
    const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
    const uint innerRowB = threadIdx.x / BN;

    // allocate thread-local cache for results in registerfile
    float threadResults[TM] = {0.0};

    // outer loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // populate the SMEM caches
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * n + innerColB];
        __syncthreads();

        // advance blocktile
        A += BK;
        B += BK * n;

        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
        // we make the dotproduct loop the outside loop, which facilitates
        // reuse of the Bs entry, which we can cache in a tmp var.
        float tmpB = Bs[dotIdx * BN + threadCol];
        for (uint resIdx = 0; resIdx < TM; ++resIdx) {
            threadResults[resIdx] +=
                As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
        }
        }
        __syncthreads();
    }

    // write out the results
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        C[(threadRow * TM + resIdx) * n + threadCol] = threadResults[resIdx];
    }
}

void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i * cols + j] = (float)rand() / RAND_MAX;
        }
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

int main(){
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C_cpu = (float*)malloc(size);
    h_C_gpu = (float*)malloc(size);

    srand(time(NULL));
    init_matrix(h_A, N, N);
    init_matrix(h_B, N, N);

    memset(h_C_cpu, 0, size);
    memset(h_C_gpu, 0, size);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C_gpu, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(CEIL_DIV(N, TILE_WIDTH), CEIL_DIV(N, TILE_WIDTH));
    dim3 dimBlock((TILE_WIDTH * TILE_WIDTH) / TM);

    printf("Matrix size: %dx%d\n", N, N);
    printf("Tile size: %dx%d\n", TILE_WIDTH, TILE_WIDTH);
    printf("TM (results per thread): %d\n", TM);
    printf("Grid size: %dx%d\n", dimGrid.x, dimGrid.y);

    printf("Performing warmup runs...\n");
    for (int i=0; i<3; i++){
        matrix_mul_cpu(h_A, h_B, h_C_cpu, N);
        block_tiling_1d_matrix_mul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
    }

    memset(h_C_cpu, 0, size);
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        matrix_mul_cpu(h_A, h_B, h_C_cpu, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        cudaMemset(d_C, 0, size);
        
        double start_time = get_time();
        block_tiling_1d_matrix_mul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    printf("CPU avg time: %.2f milliseconds\n", cpu_avg_time * 1000);
    printf("GPU avg time: %.2f milliseconds\n", gpu_avg_time * 1000);
    printf("Speedup: %.2fx\n", cpu_avg_time / gpu_avg_time);
    
    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    bool correct = true;
    int mismatches = 0;
    for (int i = 0; i < N && mismatches < 10; i++) {
        for (int j = 0; j < N && mismatches < 10; j++) {
            if (fabs(h_C_cpu[i * N + j] - h_C_gpu[i * N + j]) > 1e-3) {
                printf("Mismatch at [%d][%d]: CPU=%.6f, GPU=%.6f\n",
                       i, j, h_C_cpu[i * N + j], h_C_gpu[i * N + j]);
                mismatches++;
                correct = false;
            }
        }
    }
    
    if (correct) {
        printf("Results are correct!\n");
    } else {
        printf("Results are incorrect (%d mismatches shown)\n", mismatches);
    }
    
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
