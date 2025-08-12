#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

#define M 256
#define N 256
#define K 256
#define BLOCK_SIZE 16

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

__global__ void matrix_mul_shared_mem(float *A, float *B, float *C, int m, int k, int n){
    const int cRow = blockIdx.x;
    const int cCol = blockIdx.y;

    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    const int threadCol = threadIdx.x % BLOCK_SIZE;
    const int threadRow = threadIdx.x / BLOCK_SIZE;

    A += cRow * BLOCK_SIZE * k;
    B += cCol * BLOCK_SIZE;
    C += cRow * BLOCK_SIZE * n + cCol * BLOCK_SIZE;

    float tmp = 0.0;

    for (int bkIdx=0; bkIdx<k; bkIdx+=BLOCK_SIZE){
        As[threadRow * BLOCK_SIZE + threadCol] = A[threadRow * k + threadCol];
        Bs[threadRow * BLOCK_SIZE + threadCol] = B[threadRow * n + threadCol];

        __syncthreads();
        A += BLOCK_SIZE;
        B += BLOCK_SIZE * n;

        for (int dotIdx=0; dotIdx<BLOCK_SIZE; dotIdx++){
            tmp += As[threadRow * BLOCK_SIZE + dotIdx] * Bs[dotIdx * BLOCK_SIZE + threadCol];
        }
        __syncthreads();
    }
    C[threadRow * n + threadCol] = tmp;
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

    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
    dim3 dimGrid(ceil_div(M, BLOCK_SIZE), ceil_div(N, BLOCK_SIZE));

    printf("Performing warmup runs...\n");
    for (int i=0; i<3; i++){
        matrix_mul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        matrix_mul_shared_mem<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, K, N);
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
        matrix_mul_shared_mem<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, K, N);
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