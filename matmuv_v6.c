#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <immintrin.h>

#define N 4096
#define BLOCK_SIZE 64
#define FLOPS_PER_OP (2.0 * N * N * N)

void init_matrix(float M[N][N]);

void matmul_blocked(float A[N][N], float B[N][N], float C[N][N]) {
    // 清零结果矩阵
    memset(C, 0, sizeof(float) * N * N);
    
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
            for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
                // 处理当前块
                int i_end = (ii + BLOCK_SIZE) > N ? N : (ii + BLOCK_SIZE);
                int k_end = (kk + BLOCK_SIZE) > N ? N : (kk + BLOCK_SIZE);
                int j_end = (jj + BLOCK_SIZE) > N ? N : (jj + BLOCK_SIZE);
                
                for (int i = ii; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        __m512 a = _mm512_set1_ps(A[i][k]);
                        
                        // 内层循环处理16个元素
                        for (int j = jj; j < j_end; j += 16) {
                            __m512 c = _mm512_load_ps(&C[i][j]);
                            __m512 b = _mm512_load_ps(&B[k][j]);
                            c = _mm512_fmadd_ps(a, b, c);
                            _mm512_store_ps(&C[i][j], c);
                        }
                    }
                }
            }
        }
    }
}

int main() {
    // 分配二维数组（使用动态分配确保栈空间足够）
    float (*A)[N] = (float(*)[N])aligned_alloc(64, sizeof(float) * N * N);
    float (*B)[N] = (float(*)[N])aligned_alloc(64, sizeof(float) * N * N);
    float (*C)[N] = (float(*)[N])aligned_alloc(64, sizeof(float) * N * N);

    if (!A || !B || !C) {
        fprintf(stderr, "内存分配失败\n");
        exit(EXIT_FAILURE);
    }

    // 初始化随机数种子
    srand(time(NULL));
    
    // 初始化矩阵
    init_matrix(A);
    init_matrix(B);
    
    struct timespec start, end;
    double time_sec, gflops;
    clock_gettime(CLOCK_MONOTONIC, &start);

    matmul_blocked(A, B, C);
    clock_gettime(CLOCK_MONOTONIC, &end);

    // 计算性能
    time_sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    gflops = (FLOPS_PER_OP / time_sec) / 1e9;

    printf("矩阵维度: %d x %d\n", N, N);
    printf("块大小: %d x %d\n", BLOCK_SIZE, BLOCK_SIZE);
    printf("计算时间: %.4f 秒\n", time_sec);
    printf("性能: %.2f GFLOPS\n", gflops);
    printf("示例值: C[0][0] = %f\n", C[0][0]);

    free(A);
    free(B);
    free(C);
    return 0;
}

void init_matrix(float M[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            M[i][j] = (float)rand() / RAND_MAX * 10.0f;
        }
    }
}