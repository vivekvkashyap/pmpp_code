#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

#define FILTER_RADIUS 2
#define N 512
#define TILE_DIM 32

__constant__ float F[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i * cols + j] = (float)rand() / RAND_MAX;
        }
    }
}

void init_filter(float *filter, int size) {
    for (int i = 0; i < size; i++) {
        filter[i] = (float)rand() / RAND_MAX;
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void convolution_2d_cpu(float *I, float *P, float *filter, int r, int width, int height){
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            float Pval = 0.0;
            for (int k = 0; k < 2*r+1; k++){
                for (int l = 0; l < 2*r+1; l++){
                    int inRow = i - r + k;
                    int inCol = j - r + l;
                    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width){
                        Pval += filter[k * (2*r+1) + l] * I[inRow * width + inCol];
                    }
                }
            }
            P[i * width + j] = Pval;
        }
    }
}

__global__ void convolution_cached_tiled_2d_const_mem_kernel(float *I, float *P, int width, int height){
    int col = blockIdx.x * TILE_DIM + threadIdx.x ;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    
    __shared__ float I_s[TILE_DIM][TILE_DIM];
    if (row>=0 && row <height && col>=0 && col<width){
        I_s[threadIdx.y][threadIdx.x] = I[row * width + col];
    }
    else {
        I_s[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();
    if (col<width && row<height){
        float Pval = 0.0;
        for (int fRow=0; fRow<2*FILTER_RADIUS+1; fRow++){
            for (int fCol=0; fCol<2*FILTER_RADIUS+1; fCol++){
                if (threadIdx.x - FILTER_RADIUS + fCol<TILE_DIM  && threadIdx.y - FILTER_RADIUS + fRow<TILE_DIM){
                    Pval += F[fRow][fCol] * I_s[threadIdx.y - FILTER_RADIUS + fRow][threadIdx.x - FILTER_RADIUS + fCol];
                }
                else {
                    if (row-FILTER_RADIUS+fRow>=0 && row-FILTER_RADIUS+fRow<height && col-FILTER_RADIUS+fCol>=0 && col-FILTER_RADIUS+fCol<width){
                        Pval += F[fRow][fCol] * I[(row - FILTER_RADIUS + fRow)*width + col - FILTER_RADIUS + fCol];
                    }
                }
            }
        }
        P[row * width + col] = Pval;
    }
}

int main(){
    float  *h_I, *h_f, *h_P_cpu, *h_P_gpu;
    float *d_I, *d_P;
    size_t i_size = N * N * sizeof(float);
    size_t f_size = (2*FILTER_RADIUS+1) * (2*FILTER_RADIUS+1) * sizeof(float);

    h_I = (float*)malloc(i_size);
    h_P_cpu = (float*)malloc(i_size);
    h_P_gpu = (float*)malloc(i_size);
    h_f = (float*)malloc(f_size);

    srand(time(NULL));
    init_matrix(h_I, N, N);
    init_filter(h_f, (2*FILTER_RADIUS+1) * (2*FILTER_RADIUS+1));

    memset(h_P_cpu, 0, i_size);
    memset(h_P_gpu, 0, i_size);

    cudaMalloc(&d_I, i_size);
    cudaMalloc(&d_P, i_size);

    cudaMemcpy(d_I, h_I, i_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_P, h_P_gpu, i_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F, h_f, f_size);

    dim3 dimBlock(TILE_DIM, TILE_DIM);
    dim3 dimGrid(ceil_div(N, TILE_DIM), ceil_div(N, TILE_DIM));

    printf("Performing warmup runs...\n");
    for (int i = 0; i < 3; i++){
        convolution_2d_cpu(h_I, h_P_cpu, h_f, FILTER_RADIUS, N, N);
        convolution_cached_tiled_2d_const_mem_kernel<<<dimGrid, dimBlock>>>(d_I, d_P, N, N);
        cudaDeviceSynchronize();
    }

    memset(h_P_cpu, 0, i_size);
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        convolution_2d_cpu(h_I, h_P_cpu, h_f, FILTER_RADIUS, N, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        cudaMemset(d_P, 0, i_size);
        
        double start_time = get_time();
        convolution_cached_tiled_2d_const_mem_kernel<<<dimGrid, dimBlock>>>(d_I, d_P, N, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    printf("CPU avg time: %.2f milliseconds\n", cpu_avg_time * 1000);
    printf("GPU avg time: %.2f milliseconds\n", gpu_avg_time * 1000);
    printf("Speedup: %.2fx\n", cpu_avg_time / gpu_avg_time);
    
    cudaMemcpy(h_P_gpu, d_P, i_size, cudaMemcpyDeviceToHost);

    bool correct = true;
    int mismatches = 0;
    for (int i = 0; i < N && mismatches < 10; i++) {
        for (int j = 0; j < N && mismatches < 10; j++) {
            if (fabs(h_P_cpu[i * N + j] - h_P_gpu[i * N + j]) > 1e-3) {
                printf("Mismatch at [%d][%d]: CPU=%.6f, GPU=%.6f\n",
                       i, j, h_P_cpu[i * N + j], h_P_gpu[i * N + j]);
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
    
    free(h_I);
    free(h_f);
    free(h_P_cpu);
    free(h_P_gpu);
    cudaFree(d_I);
    cudaFree(d_P);
    
    return 0;
}