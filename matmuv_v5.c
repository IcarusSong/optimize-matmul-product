#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
// #include <omp.h>
#include <immintrin.h>

#define N 4096
#define BLOCK_SIZE 64
#define FLOPS_PER_OP (2.0 * N * N * N)

#define _A(i, j) A[(i) * N + (j)]
#define _B(i, j) B[(i) * N + (j)]
#define _C(i, j) C[(i) * N + (j)]
#define _M(i, j) M[(i) * N + (j)]

void init_matrix(float*);

void matmul_blocked(float* A, float* B, float* C) {
    // 确保BLOCK_SIZE是16的倍数，因为AVX-512处理16个float
    // static_assert(BLOCK_SIZE % 16 == 0, "BLOCK_SIZE must be multiple of 16 for AVX-512");
    
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
            for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
                // 处理当前块
                int i_end = (ii + BLOCK_SIZE) > N ? N : (ii + BLOCK_SIZE);
                int k_end = (kk + BLOCK_SIZE) > N ? N : (kk + BLOCK_SIZE);
                int j_end = (jj + BLOCK_SIZE) > N ? N : (jj + BLOCK_SIZE);
                
                for (int i = ii; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        __m512 a = _mm512_set1_ps(_A(i, k));
                        
                        // 内层循环处理16个元素
                        for (int j = jj; j < j_end; j += 16) {
                            __m512 c = _mm512_load_ps(&_C(i, j));
                            __m512 b = _mm512_load_ps(&_B(k, j));
                            c = _mm512_fmadd_ps(a, b, c);
                            _mm512_store_ps(&_C(i, j), c);
                        }
                    }
                }
            }
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

    matmul_blocked(A, B, C);
    clock_gettime(CLOCK_MONOTONIC, &end);

    // 计算性能
    time_sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    gflops = (FLOPS_PER_OP / time_sec) / 1e9;

    printf("矩阵维度: %d x %d\n", N, N);
    printf("块大小: %d x %d\n", BLOCK_SIZE, BLOCK_SIZE);
    printf("计算时间: %.4f 秒\n", time_sec);
    printf("性能: %.2f GFLOPS\n", gflops);
    printf("示例值: C[0][0] = %f\n", C[0]);

    free(A);
    free(B);
    free(C);
    return 0;
}

void init_matrix(float* M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            _M(i, j) = (float)rand() / RAND_MAX * 10.0f;
        }
    }
}