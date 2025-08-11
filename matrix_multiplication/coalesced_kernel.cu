#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>


#define N 256
#define BLOCK_SIZE 16  

void matrix_mul_cpu(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

__global__ void coalesced_matrix_mul(float *A, float *B, float *C, int n) {
    const int x = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
    const int y = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);
    
    if (x < n && y < n) {
        float tmp = 0.0f;
        for (int i = 0; i < n; ++i) {
            tmp += A[x * n + i] * B[i * n + y];
        }
        C[x * N + y] = tmp;
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
    
    int threads_per_block = BLOCK_SIZE * BLOCK_SIZE; 
    dim3 dimBlock(threads_per_block, 1, 1);  
    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);  // 2D grid
    
    printf("Block size: %d threads (1D), Grid size: %dx%d\n", 
           threads_per_block, dimGrid.x, dimGrid.y);
    printf("BLOCK_SIZE: %d, Matrix size: %dx%d\n", BLOCK_SIZE, N, N);
    
    printf("Performing warmup runs...\n");
    for (int i = 0; i < 3; i++) {
        matrix_mul_cpu(h_A, h_B, h_C_cpu, N);
        coalesced_matrix_mul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
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
        coalesced_matrix_mul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
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