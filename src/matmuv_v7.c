#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <immintrin.h>

#define N 4096
#define L1_BLOCK_SIZE 64
#define L2_BLOCK_SIZE 512
#define FLOPS_PER_OP (2.0 * N * N * N)
#define PREFETCH_DISTANCE 128

void init_matrix(float* M);

void mat_mul_blocked(float* A, float* B, float* C) {
    memset(C, 0, sizeof(float) * N * N);

    for (int l2_i = 0; l2_i < N; l2_i += L2_BLOCK_SIZE) {
        for (int l2_j = 0; l2_j < N; l2_j += L2_BLOCK_SIZE) {
            for (int l2_k = 0; l2_k < N; l2_k += L2_BLOCK_SIZE) {
                
                for (int l1_i = l2_i; l1_i < l2_i + L2_BLOCK_SIZE; l1_i += L1_BLOCK_SIZE) {
                    for (int l1_j = l2_j; l1_j < l2_j + L2_BLOCK_SIZE; l1_j += L1_BLOCK_SIZE) {
                        for (int l1_k = l2_k; l1_k < l2_k + L2_BLOCK_SIZE; l1_k += L1_BLOCK_SIZE) {
                            
                            for (int i = l1_i; i < l1_i + L1_BLOCK_SIZE && i < N; i++) {
                                float* a_row = &A[i * N];
                                float* c_row = &C[i * N];
                                
                                for (int j = l1_j; j < l1_j + L1_BLOCK_SIZE && j < N; j += 32) {
                                    __m512 c_vec1 = _mm512_load_ps(c_row + j);
                                    __m512 c_vec2 = _mm512_load_ps(c_row + j + 16);
                                    
                                    for (int k = l1_k; k < l1_k + L1_BLOCK_SIZE && k < N; k++) {
                                        float* b_row = &B[k * N];
                                        
                                        // 预取
                                        if (k + PREFETCH_DISTANCE < l1_k + L1_BLOCK_SIZE) {
                                            _mm_prefetch((char*)&A[i * N + k + PREFETCH_DISTANCE], _MM_HINT_T0);
                                            _mm_prefetch((char*)&B[(k + PREFETCH_DISTANCE) * N + j], _MM_HINT_T0);
                                            _mm_prefetch((char*)&B[(k + PREFETCH_DISTANCE) * N + j + 16], _MM_HINT_T0);
                                        }
                                        
                                        __m512 a = _mm512_set1_ps(a_row[k]);
                                        __m512 b1 = _mm512_load_ps(b_row + j);
                                        __m512 b2 = _mm512_load_ps(b_row + j + 16);
                                        c_vec1 = _mm512_fmadd_ps(a, b1, c_vec1);
                                        c_vec2 = _mm512_fmadd_ps(a, b2, c_vec2);
                                    }
                                    
                                    _mm512_store_ps(c_row + j, c_vec1);
                                    _mm512_store_ps(c_row + j + 16, c_vec2);
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
    float* A = (float*)aligned_alloc(64, N * N * sizeof(float));
    float* B = (float*)aligned_alloc(64, N * N * sizeof(float));
    float* C = (float*)aligned_alloc(64, N * N * sizeof(float));

    if (!A || !B || !C) {
        fprintf(stderr, "内存分配失败\n");
        exit(EXIT_FAILURE);
    }

    srand(time(NULL));
    init_matrix(A);
    init_matrix(B);
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    mat_mul_blocked(A, B, C);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double gflops = (FLOPS_PER_OP / time_sec) / 1e9;

    printf("矩阵维度: %d x %d\n", N, N);
    printf("L1块大小: %d\n", L1_BLOCK_SIZE);
    printf("L2块大小: %d\n", L2_BLOCK_SIZE);
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
            M[i * N + j] = (float)rand() / RAND_MAX * 10.0f;
        }
    }
}