#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "include/errors.h"
#include "include/lin_alg.h"

// Globals for the library
cublasHandle_t handle;
cusolverDnHandle_t solver_handle;

void init_cuda() {
    cublasCreate(&handle);
    cusolverDnCreate(&solver_handle);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
}

void test_vec_creation_and_zeroing() {
    size_t n = 100;
    vector_t *vec = lin_alg_create_vector(n);
    float *h_data = (float*)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) h_data[i] = 123.45f;

    cudaMemcpy(vec->vector, h_data, n * sizeof(float), cudaMemcpyHostToDevice);
    lin_alg_zero_vector(vec);
    cudaMemcpy(h_data, vec->vector, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < n; i++) assert(h_data[i] == 0.0f);

    free(h_data);
    lin_alg_free_vector(&vec);
}

void test_vec_dot_product() {
    size_t n = 10;
    vector_t *v1 = lin_alg_create_vector(n);
    vector_t *v2 = lin_alg_create_vector(n);
    float h1[10], h2[10];

    for (size_t i = 0; i < n; i++) {
        h1[i] = 1.0f;
        h2[i] = (float)(i + 1);
    }

    cudaMemcpy(v1->vector, h1, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(v2->vector, h2, n * sizeof(float), cudaMemcpyHostToDevice);

    float res = 0.0f;
    lin_alg_dot_productv(v1, v2, &res);
    assert(fabsf(res - 55.0f) < 1e-5f);

    lin_alg_free_vector(&v1);
    lin_alg_free_vector(&v2);
}

void test_qr_decomposition() {
    uint16_t n = 4;
    matrix_t *mtx = lin_alg_create_matrix(n, n);
    
    // Fill with random normal values using your GPU function
    lin_alg_normal_rand_matrix(mtx);
    
    // Decompose
    uint8_t status = lin_alg_qr_decompose(mtx);
    assert(status == MATH_OPS_SUCCESS);

    // Verify orthogonality Q^T * Q = I
    // Since everything is on GPU, we create a result matrix to hold Q^T * Q
    matrix_t *res_mtx = lin_alg_create_matrix(n, n);
    float alpha = 1.0f, beta = 0.0f;
    
    // Multiply Q^T * Q
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &alpha, 
                mtx->matrix, mtx->stride, mtx->matrix, mtx->stride, 
                &beta, res_mtx->matrix, res_mtx->stride);

    float *h_res = (float*)malloc(n * res_mtx->stride * sizeof(float));
    cudaMemcpy(h_res, res_mtx->matrix, n * res_mtx->stride * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float val = h_res[j * res_mtx->stride + i];
            if (i == j) assert(fabsf(val - 1.0f) < 1e-3f);
            else assert(fabsf(val) < 1e-3f);
        }
    }

    free(h_res);
    lin_alg_free_matrix(&mtx);
    lin_alg_free_matrix(&res_mtx);
}

int main() {
    init_cuda();
    
    test_vec_creation_and_zeroing();
    printf("test_vec_creation_and_zeroing(): success!\n");

    test_vec_dot_product();
    printf("test_vec_dot_product(): success!\n");

    test_qr_decomposition();
    printf("test_qr_decomposition(): success!\n");

    return 0;
}
