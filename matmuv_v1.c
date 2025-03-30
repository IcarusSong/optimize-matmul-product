#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>

#define N 1024
#define BLOCK_SIZE 64
#define FLOPS_PER_OP (2.0 * N * N * N)

#define _A(i, j) A[(i) * N + (j)]
#define _B(i, j) B[(i) * N + (j)]
#define _C(i, j) C[(i) * N + (j)]
#define _M(i, j) M[(i) * N + (j)]

void init_matrix(float*);

void matmul(float* A, float* B, float* C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += _A(i, k) * _B(k, j);
            }
            _C(i, j) = sum;
        }
    }
}


int main() {
    // 分配对齐内存
    float* A = (float*)aligned_alloc(64, sizeof(float) * N * N);
    float* B = (float*)aligned_alloc(64, sizeof(float) * N * N);
    float* C = (float*)aligned_alloc(64, sizeof(float) * N * N);

    if (!A || !B || !C) {
        fprintf(stderr, "内存分配失败\n");
        exit(EXIT_FAILURE);
    }

    // 初始化随机数种子
    srand(time(NULL));
    
    // 初始化矩阵
    init_matrix(A);
    init_matrix(B);
    memset(C, 0, N*N*sizeof(float));
    
    struct timespec start, end;
    double time_sec, gflops;
    clock_gettime(CLOCK_MONOTONIC, &start);

    matmul(A, B, C);
    clock_gettime(CLOCK_MONOTONIC, &end);

    // 计算性能
    time_sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    gflops = (FLOPS_PER_OP / time_sec) / 1e9;

    printf("矩阵维度: %d x %d\n", N, N);
    printf("计算时间: %.4f 秒\n", time_sec);
    printf("性能: %.2f GFLOPS\n", gflops);
    printf("示例值: C[0][0] = %f\n", C[0]);

    memset(C, 0, N*N*sizeof(float));
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    clock_gettime(CLOCK_MONOTONIC, &end);

    time_sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    gflops = (FLOPS_PER_OP / time_sec) / 1e9;

    free(A);
    free(B);
    free(C);

}

void init_matrix(float* M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            _M(i, j) = (float)rand() / RAND_MAX * 10.0f;
        }
    }
}