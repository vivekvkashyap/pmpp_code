#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

#define CHANNELS 3
#define BLUR_SIZE 4

__global__ void blur_gpu(unsigned char *d_blur, unsigned char *d_color, int width, int height, int channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        for (int c = 0; c < channels; c++) {
            int pixval = 0;
            int pixels = 0;

            for (int i = -BLUR_SIZE; i <= BLUR_SIZE; i++) {
                for (int j = -BLUR_SIZE; j <= BLUR_SIZE; j++) {
                    int curRow = row + i;
                    int curCol = col + j;

                    if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                        int pixel_index = (curRow * width + curCol) * channels + c;
                        pixval += d_color[pixel_index];
                        ++pixels;
                    }
                }
            }

            int out_index = (row * width + col) * channels + c;
            d_blur[out_index] = (unsigned char)(pixval / pixels);
        }
    }
}

void save_image_jpg(unsigned char* image_data, int width, int height, int channels, const char* filename) {
    int result = stbi_write_jpg(filename, width, height, channels, image_data, 90); 
    if (result) {
        printf("Image saved successfully: %s\n", filename);
    } else {
        printf("Failed to save image\n");
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    unsigned char *h_blur_gpu;
    unsigned char *d_color, *d_blur;

    int width, height, channels;
    unsigned char *h_color = stbi_load("f1_ferrari.jpg", &width, &height, &channels, 0);
    if (!h_color) {
        printf("Failed to load image\n");
        return -1;
    }

    printf("Image loaded: %d x %d pixels, %d channels\n", width, height, channels);

    size_t color_size = width * height * channels * sizeof(unsigned char);

    h_blur_gpu = (unsigned char*)malloc(color_size);

    cudaMalloc(&d_color, color_size);
    cudaMalloc(&d_blur, color_size);

    cudaMemcpy(d_color, h_color, color_size, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
                 (height + dimBlock.y - 1) / dimBlock.y,
                 1);

    printf("Performing warmup runs...\n");
    for (int i = 0; i < 3; i++) {
        blur_gpu<<<dimGrid, dimBlock>>>(d_blur, d_color, width, height, channels);
        cudaDeviceSynchronize();
    }

    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        blur_gpu<<<dimGrid, dimBlock>>>(d_blur, d_color, width, height, channels);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += (end_time - start_time);
    }
    double gpu_avg_time = gpu_total_time / 20.0;
    printf("GPU avg time: %.3f milliseconds\n", gpu_avg_time * 1000);

    cudaMemcpy(h_blur_gpu, d_blur, color_size, cudaMemcpyDeviceToHost);

    save_image_jpg(h_blur_gpu, width, height, 3, "output_gray.jpg");

    stbi_image_free(h_color);
    free(h_blur_gpu);
    cudaFree(d_color);
    cudaFree(d_blur);

    return 0;
}
