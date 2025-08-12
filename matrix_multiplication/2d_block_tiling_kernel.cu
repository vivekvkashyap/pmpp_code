#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

#define M 256
#define N 256
#define K 256

#define BM 64
#define BN 64
#define BK 8
#define TM 8
#define TN 8


int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

void matrix_mul_cpu(float *A, float *B, float *C, int m, int k, int n){
    for (int i=0; i<m; i++){
        for (int j=0; j<n; j++){
            float sum = 0.0f;
            for (int l=0; l<k; l++){
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

__global__ void block_tiling_2d_matrix_mul(float *A, float *B, float *C, int m, int k, int n){
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const uint totalResultsBlocktile = BM * BN;
    const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

    const uint threadCol = threadIdx.x % (BN / TN);
    const uint threadRow = threadIdx.x / (BN / TN); 

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * k;
    B += cCol * BN;
    C += cRow * BM * n + cCol * BN;

    const uint innerColA = threadIdx.x % BK;
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN;
    const uint innerRowB = threadIdx.x / BN;

    const uint strideA = numThreadsBlocktile / BK;
    const uint strideB = numThreadsBlocktile / BN;

    float threadResults[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for (uint bkIdx=0; bkIdx<k; bkIdx+=BK){
        for (uint loadOffset=0; loadOffset<BM; loadOffset+=strideA){
            As[(innerRowA + loadOffset) * BK + innerColA] = A[(innerRowA + loadOffset) * k + innerColA];
        }

        for (uint loadOffset=0; loadOffset<BK; loadOffset+=strideB){
            Bs[(innerRowB + loadOffset) * BN + innerColB] = B[(innerRowB + loadOffset) * n + innerColB];
        }

        __syncthreads();
 
        A += BK;   
        B += BK * N;

        for (uint dotIdx=0; dotIdx<BK; dotIdx++){
            for (uint i=0; i<TM; i++){
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }

            for (uint i=0; i<TN; i++){
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }

            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                threadResults[resIdxM * TN + resIdxN] +=
                    regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] = threadResults[resIdxM * TN + resIdxN];
        }
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

int main(){
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    float *d_A, *d_B, *d_C;
    size_t a_size = M * K * sizeof(float);
    size_t b_size = K * N * sizeof(float);
    size_t c_size = M * N * sizeof(float);

    h_A = (float*)malloc(a_size);
    h_B = (float*)malloc(b_size);
    h_C_cpu = (float*)malloc(c_size);
    h_C_gpu = (float*)malloc(c_size);

    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    memset(h_C_cpu, 0, c_size);
    memset(h_C_gpu, 0, c_size);

    cudaMalloc(&d_A, a_size);
    cudaMalloc(&d_B, b_size);
    cudaMalloc(&d_C, c_size);

    cudaMemcpy(d_A, h_A, a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, b_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C_gpu, c_size, cudaMemcpyHostToDevice);

    dim3 dimBlock((BM * BN)/ (TM * TN));
    dim3 dimGrid(ceil_div(N, BN), ceil_div(M, BM));

    printf("Performing warmup runs...\n");
    for (int i=0; i<3; i++){
        matrix_mul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        block_tiling_2d_matrix_mul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
    }

    memset(h_C_cpu, 0, c_size);
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        matrix_mul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        cudaMemset(d_C, 0, c_size);
        
        double start_time = get_time();
        block_tiling_2d_matrix_mul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    printf("CPU avg time: %.2f milliseconds\n", cpu_avg_time * 1000);
    printf("GPU avg time: %.2f milliseconds\n", gpu_avg_time * 1000);
    printf("Speedup: %.2fx\n", cpu_avg_time / gpu_avg_time);
    
    cudaMemcpy(h_C_gpu, d_C, c_size, cudaMemcpyDeviceToHost);

    bool correct = true;
    int mismatches = 0;
    for (int i = 0; i < M && mismatches < 10; i++) {
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