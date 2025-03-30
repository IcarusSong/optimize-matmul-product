#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>

#define N 4096
#define L2_BLOCK_SIZE 256
#define L1_BLOCK_SIZE 64
#define FLOPS_PER_OP (2.0 * N * N * N)

// 分配二维数组
float** allocate_matrix() {
    float** mat = (float**)malloc(N * sizeof(float*));
    if (!mat) return NULL;
    
    for (int i = 0; i < N; i++) {
        if (posix_memalign((void**)&mat[i], 64, N * sizeof(float))) {
            fprintf(stderr, "内存分配失败\n");
            for (int j = 0; j < i; j++) free(mat[j]);
            free(mat);
            return NULL;
        }
    }
    return mat;
}

// 释放二维数组
void free_matrix(float** mat) {
    if (!mat) return;
    for (int i = 0; i < N; i++) {
        free(mat[i]);
    }
    free(mat);
}

// 初始化矩阵
void init_matrix(float** mat) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            mat[i][j] = (float)rand() / RAND_MAX * 10.0f;
        }
    }
}

void mat_mul_blocked(float** A, float** B, float** C) {

    const int prefetch_distance = 64;

    // 进行L2分块
    for (int l2_i = 0; l2_i < N; l2_i += L2_BLOCK_SIZE) {
        for (int l2_j = 0; l2_j < N; l2_j += L2_BLOCK_SIZE) {
            for (int l2_k = 0; l2_k < N; l2_k += L2_BLOCK_SIZE) {
                // 进行L1分块
                for (int l1_i = l2_i; l1_i < l2_i + L2_BLOCK_SIZE; l1_i += L1_BLOCK_SIZE) {
                    for (int l1_j = l2_j; l1_j < l2_j + L2_BLOCK_SIZE; l1_j += L1_BLOCK_SIZE) {
                        for (int l1_k = l2_k; l1_k < l2_k + L2_BLOCK_SIZE; l1_k += L1_BLOCK_SIZE) {
                            // 计算当前L1分块
                            for (int i = l1_i; i < l1_i + L1_BLOCK_SIZE && i < N; i++) {
                                for (int j = l1_j; j < l1_j + L1_BLOCK_SIZE && j < N; j += 32) {

                                    __m512 c_vec1 = _mm512_load_ps(&C[i][j]);
                                    __m512 c_vec2 = _mm512_load_ps(&C[i][j + 16]);
                                                                            // 预取下一轮A和B的数据
                                    // if (l1_k + prefetch_distance < l2_k + L2_BLOCK_SIZE) {
                                    //     _mm_prefetch(&A[i][l1_k + prefetch_distance], _MM_HINT_T0);
                                    //     _mm_prefetch(&B[l1_k + prefetch_distance][j], _MM_HINT_T0);
                                    // }

                                    // 累加
                                    for (int k = l1_k; k < l1_k + L1_BLOCK_SIZE && k < N - 1; k+=2) {
                                        __m512 a1 = _mm512_set1_ps(A[i][k]);
                                        __m512 a2 = _mm512_set1_ps(A[i][k + 1]);

                                        __m512 b1_row1 = _mm512_load_ps(&B[k][j]);
                                        __m512 b1_row2 = _mm512_load_ps(&B[k][j + 16]);
                                        __m512 b2_row1 = _mm512_load_ps(&B[k + 1][j]);
                                        __m512 b2_row2 = _mm512_load_ps(&B[k + 1][j + 16]);

                                        c_vec1 = _mm512_fmadd_ps(a1, b1_row1, c_vec1);
                                        c_vec2 = _mm512_fmadd_ps(a1, b1_row2, c_vec2);
                                        c_vec1 = _mm512_fmadd_ps(a2, b2_row1, c_vec1);
                                        c_vec2 = _mm512_fmadd_ps(a2, b2_row2, c_vec2);
                                    }
                                    _mm512_store_ps(&C[i][j], c_vec1);
                                     _mm512_store_ps(&C[i][j + 16], c_vec2);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

int main() {
    // 分配二维数组
    float** A = allocate_matrix()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      ;
    float** B = allocate_matrix();
    float** C = allocate_matrix();
    
    if (!A || !B || !C) {
        fprintf(stderr, "内存分配失败\n");
        exit(EXIT_FAILURE);
    }

    // 初始化随机数种子
    srand(time(NULL));
    
    // 初始化矩阵
    init_matrix(A);
    init_matrix(B);
    for (int i = 0; i < N; i++) {
        memset(C[i], 0, N * sizeof(float));
    }
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    mat_mul_blocked(A, B, C);
    clock_gettime(CLOCK_MONOTONIC, &end);

    // 计算性能
    double time_sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double gflops = (FLOPS_PER_OP / time_sec) / 1e9;

    printf("矩阵维度: %d x %d\n", N, N);
    printf("L1块大小: %d\n", L1_BLOCK_SIZE);
    printf("L2块大小: %d\n", L2_BLOCK_SIZE);
    printf("计算时间: %.4f 秒\n", time_sec);
    printf("性能: %.2f GFLOPS\n", gflops);
    printf("示例值: C[0][0] = %f\n", C[0][0]);

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);

    return 0;
}
