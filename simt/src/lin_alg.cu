#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cusolverDn.h>
#include "../include/lin_alg.h"
#include "../include/errors.h"

#define TRANSPOSE_FLAG(n) (cublasOperation_t) n

static cublasHandle_t handle = NULL;
static cusolverDnHandle_t solver_handle = NULL;

static int lin_alg_init_cublas(void) {
    if (handle != NULL)
        return SUCCESS;

    if (cublasCreate_v2(&handle) != CUBLAS_STATUS_SUCCESS)
        return ERROR;

    return SUCCESS;
}

static int lin_alg_init_cusolver(void) {
    if (solver_handle != NULL)
        return SUCCESS;

    if (cusolverDnCreate(&solver_handle) != CUSOLVER_STATUS_SUCCESS)
        return ERROR;

    return SUCCESS;
}

int lin_alg_runtime_init(void) {
    if (lin_alg_init_cublas() != SUCCESS)
        return ERROR;

    if (lin_alg_init_cusolver() != SUCCESS) {
        cublasDestroy_v2(handle);
        handle = NULL;
        return ERROR;
    }

    return SUCCESS;
}

int lin_alg_runtime_shutdown(void) {
    int status = SUCCESS;

    if (solver_handle != NULL) {
        if (cusolverDnDestroy(solver_handle) != CUSOLVER_STATUS_SUCCESS)
            status = ERROR;
        solver_handle = NULL;
    }

    if (handle != NULL) {
        if (cublasDestroy_v2(handle) != CUBLAS_STATUS_SUCCESS)
            status = ERROR;
        handle = NULL;
    }

    return status;
}

int lin_alg_set_stream(void *stream_handle) {
    if (stream_handle == NULL)
        return ERROR;

    if (lin_alg_runtime_init() != SUCCESS)
        return ERROR;

    cudaStream_t stream = (cudaStream_t)stream_handle;

    if (cublasSetStream_v2(handle, stream) != CUBLAS_STATUS_SUCCESS)
        return ERROR;
    if (cusolverDnSetStream(solver_handle, stream) != CUSOLVER_STATUS_SUCCESS)
        return ERROR;

    return SUCCESS;
}

/* TODO: Implement separation of concern between memory management and processing */

matrix_t* lin_alg_create_matrix(const size_t m, const size_t n) {
    if (m == 0 || n == 0)
        return NULL;

    // 1. Allocate the struct on the HOST (CPU)
    matrix_t *matrix_var = (matrix_t*)malloc(sizeof(matrix_t));
    if (matrix_var == NULL)
        return NULL;

    // 2. Set metadata on the Host
    matrix_var->m = m;
    matrix_var->n = n;
    matrix_var->transpose_flag = CUBLAS_OP_N;
    matrix_var->stride = (m + 31) & ~31; // Good alignment logic!
    size_t total_bytes = n * (matrix_var->stride) * sizeof(float);

    // 3. Allocate the actual data on the DEVICE (GPU)
    if (cudaMalloc(&matrix_var->matrix, total_bytes) != cudaSuccess) {
        free(matrix_var); // Free the host struct
        return NULL;
    }

    // 4. Initialize GPU memory to zero
    if (cudaMemset(matrix_var->matrix, 0, total_bytes) != cudaSuccess) {
        cudaFree(matrix_var->matrix);
        free(matrix_var);
        return NULL;
    }

    return matrix_var;
}

int lin_alg_free_matrix(matrix_t **matrix) {
    if (matrix == NULL)
        return ERROR;

    if (*matrix) {
        if ((*matrix)->matrix)
            cudaFree((*matrix)->matrix);
        free(*matrix);
        *matrix = NULL;
    }

    return SUCCESS;
}

vector_t* lin_alg_create_vector(const size_t n) {
    if (n == 0)
        return NULL;

    vector_t *vector = NULL;

    if ((vector = (vector_t*) malloc(sizeof(vector_t))) == NULL)
        return NULL;

    vector->n = n;

    if (cudaMalloc(&vector->vector, n * sizeof(float)) != cudaSuccess) {
        free(vector);
        return NULL;
    }

    return vector;
}

int lin_alg_free_vector(vector_t ** vector) {
    if (vector == NULL)
        return ERROR;

    if (*vector) {
        if ((*vector)->vector)
            cudaFree((*vector)->vector);

        free(*vector);
        *vector = NULL;
    }

    return SUCCESS;
}

/* Zero vector / clear a vector */
int lin_alg_zero_vector(vector_t *vector) {
    if (vector == NULL)
        return ERROR;
    else if (vector->vector == NULL || vector->n == 0)
        return ERROR;

    if (cudaMemset(vector->vector, 0, vector->n * sizeof(float)) != cudaSuccess)
        return ERROR;

    return SUCCESS;
}

/* Copy vec_2 into vec_1 */
int lin_alg_copy_vector(vector_t * vector_dest, const vector_t *vector_src) {
    if (vector_dest == NULL || vector_src == NULL)
        return ERROR;
    else if (vector_dest->n == 0 || vector_src->n == 0 ||
            vector_dest->n != vector_src->n)
        return ERROR;
    else if (vector_dest->vector == NULL || vector_src->vector == NULL)
        return ERROR;

    if (lin_alg_runtime_init() != SUCCESS)
        return ERROR;

    if (cublasScopy(handle, vector_dest->n, vector_src->vector, 1,
                vector_dest->vector, 1) != CUBLAS_STATUS_SUCCESS)
        return ERROR;

    return SUCCESS;
}

int lin_alg_copy_matrix(matrix_t *matrix_dest, const matrix_t *matrix_src) {
    if (matrix_dest == NULL || matrix_src == NULL)
        return ERROR;
    else if (matrix_dest->matrix == NULL || matrix_src->matrix == NULL ||
            matrix_dest->m != matrix_src->m || matrix_dest->n != matrix_src->n)
        return ERROR;

    if (lin_alg_runtime_init() != SUCCESS)
        return ERROR;

    float alpha = 1.0f, beta = 0.0f;

    if (cublasSgeam(handle, TRANSPOSE_FLAG(matrix_src->transpose_flag), TRANSPOSE_FLAG(matrix_dest->transpose_flag),
                matrix_src->m, matrix_src->n, &alpha, matrix_src->matrix, matrix_src->stride, &beta,
                matrix_src->matrix, matrix_src->stride, matrix_dest->matrix, matrix_dest->stride) != CUBLAS_STATUS_SUCCESS)
        return ERROR;

    return SUCCESS;
}

/* Add vec_1 to vec_2, store result in vec_1 */
/* Result: vector_dest = (1.0 * vector_src) + vector_dest */
int lin_alg_add_vectors(vector_t *vector_dest, const vector_t *vector_src) {
    if (vector_dest == NULL || vector_src == NULL)
        return ERROR;
    else if (vector_dest->n != vector_src->n || vector_src->vector == NULL ||
            vector_dest->vector == NULL)
        return ERROR;

    if (lin_alg_runtime_init() != SUCCESS)
        return ERROR;

    const float alpha = 1.0f;

    // cublasSaxpy(handle, n, alpha, x, incx, y, incy)
    if (cublasSaxpy(handle,
                    vector_dest->n,
                    &alpha,
                    vector_src->vector, 1,
                    vector_dest->vector, 1) != CUBLAS_STATUS_SUCCESS) {
        return ERROR;
    }

    return SUCCESS;
}

/* Substract vec_2 from vec_1, store result in vec_1 */
int lin_alg_sub_vectors(vector_t *vector_dest, const vector_t *vector_src) {
    if (vector_dest == NULL || vector_src == NULL)
        return ERROR;
    else if (vector_dest->n != vector_src->n || vector_dest->vector == NULL ||
            vector_src->vector == NULL)
        return ERROR;

    if (lin_alg_runtime_init() != SUCCESS)
        return ERROR;

    const float alpha = -1.0f;

    // cublasSaxpy(handle, n, alpha, x, incx, y, incy)
    if (cublasSaxpy(handle,
                    vector_dest->n,
                    &alpha,
                    vector_src->vector, 1,
                    vector_dest->vector, 1) != CUBLAS_STATUS_SUCCESS) {
        return ERROR;
    }

    return SUCCESS;
}

/* Scale vec with constant α, store result in vec */
int lin_alg_scale_vector(vector_t *vector, const float α) {
    if (vector == NULL)
        return ERROR;
    else if (vector->vector == NULL || vector->n == 0)
        return ERROR;

    if (lin_alg_runtime_init() != SUCCESS)
        return ERROR;

    if (cublasSscal(handle, vector->n, &α, vector->vector, 1) != CUBLAS_STATUS_SUCCESS)
        return ERROR;

    return SUCCESS;
}

/* Dot product of vec_1 and vec_2, return result */
int lin_alg_dot_productv(const vector_t *vec_1, const vector_t* vec_2,
        float *result) {
    if (vec_1 == NULL || vec_2 == NULL || result == NULL)
        return ERROR;
    else if (vec_1->vector == NULL || vec_2->vector == NULL ||
            vec_1->n != vec_2->n)
        return ERROR;

    if (lin_alg_runtime_init() != SUCCESS)
        return ERROR;

    if (cublasSdot(handle, vec_1->n, vec_1->vector, 1, vec_2->vector, 1,
                result) != CUBLAS_STATUS_SUCCESS)
        return ERROR;

    return SUCCESS;
}

/* Dot product of mtx and vec, store result in res */
int lin_alg_dot_productmv(const matrix_t *matrix, const vector_t *vector,
        vector_t *result) {
    if (matrix == NULL || vector == NULL || result == NULL)
        return ERROR;
    else if (matrix->matrix == NULL || vector->vector == NULL ||
            result->vector == NULL || matrix->n != vector->n)
        return ERROR;

    if (lin_alg_runtime_init() != SUCCESS)
        return ERROR;

    float alpha = 1.0f;
    float beta = 0.0f;

    if (cublasSgemv(handle, TRANSPOSE_FLAG(matrix->transpose_flag), matrix->m, matrix->n,
                &alpha, matrix->matrix, matrix->stride, vector->vector, 1,
                &beta, result->vector, 1) != CUBLAS_STATUS_SUCCESS)
        return ERROR;

    return SUCCESS;
}

/* Transpose a matrix mtx */
matrix_t* lin_alg_transpose_matrix(matrix_t *matrix) {
    if (matrix == NULL)
        return NULL;
    else if (matrix->matrix == NULL || matrix->m == 0
            || matrix->n == 0)
        return NULL;

    if (TRANSPOSE_FLAG(matrix->transpose_flag) == CUBLAS_OP_N)
        matrix->transpose_flag = (uint8_t) CUBLAS_OP_T;
    else if (TRANSPOSE_FLAG(matrix->transpose_flag) == CUBLAS_OP_T)
        matrix->transpose_flag = (uint8_t) CUBLAS_OP_N;

    return matrix;
}

/* Computing the l2 norm of a vector */
float lin_alg_l2(const vector_t *vec) {
    if (vec == NULL)
        return -1.0f;
    else if (vec->vector == NULL || vec->n == 0)
        return -1.0f;

    if (lin_alg_runtime_init() != SUCCESS)
        return -1.0f;

    float result = 0.0f;

    if (cublasSnrm2(handle, vec->n, vec->vector, 1, &result) != CUBLAS_STATUS_SUCCESS)
        return -1.0f;

    return result;
}

/* TODO: Extend the input from uint16_t to uint32_t */
/* Calculate the L2 norm of input vectors to ensure length is 1,
 * projecting the vector on the hypersphere S^d-1 */
uint8_t lin_alg_l2_normalize(vector_t *vec) {
    if (vec == NULL)
        return MATH_OPS_NULL;
    else if (vec->vector == NULL || vec->n == 0)
        return MATH_OPS_EMPTY;

    float l2_norm = lin_alg_l2(vec);
    if (l2_norm < 0.0f)
        return MATH_L2_FAILED;

    if (l2_norm < EPSILON) return MATH_L2_FAILED;

    float scaler = 1.0f / l2_norm;
    if (lin_alg_scale_vector(vec, scaler) != SUCCESS)
        return MATH_L2_FAILED;

    return MATH_OPS_SUCCESS;
}

/* Generate a square nxn matrix where each row is random normal distribution values */
matrix_t* lin_alg_normal_rand_matrix(matrix_t *mtx) {
    if (mtx == NULL)
        return NULL;
    else if (mtx->matrix == NULL || mtx->m == 0 || mtx->n == 0 ||
            mtx->m != mtx->n)
        return NULL;

    // 2. Create cuRAND generator
    curandGenerator_t gen;
    if (curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS)
        return NULL;

    // 3. Set seed (using current time or a fixed value)
    if (curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) != CURAND_STATUS_SUCCESS) {
        curandDestroyGenerator(gen);
        return NULL;
    }

    /*
       4. Generate Normal Distribution
       Mean: 0.0f, StdDev: 1.0f
       Total elements to generate: n * stride
       (We fill the stride/padding to keep it a simple contiguous call)
    */
    size_t total_elements = mtx->n * mtx->stride;
    if (curandGenerateNormal(gen, mtx->matrix, total_elements, 0.0f, 1.0f) != CURAND_STATUS_SUCCESS) {
        curandDestroyGenerator(gen);
        lin_alg_free_matrix(&mtx);
        return NULL;
    }

    // 5. Cleanup
    curandDestroyGenerator(gen);

    return mtx;
}

/* Apply QR decomposition to ensure orthogonality.
   This replaces the manual Gram-Schmidt with an optimized Householder QR. */
uint8_t lin_alg_qr_decompose(matrix_t *matrix) {
    if (matrix == NULL || matrix->matrix == NULL)
        return MATH_OPS_NULL;
    if (matrix->m != matrix->n)
        return MATH_QR_FAILED; // cuSOLVER handles non-square, but API expects square

    if (lin_alg_runtime_init() != SUCCESS)
        return MATH_QR_FAILED;

    int m = (int)matrix->m;
    int n = (int)matrix->n;
    int lda = (int)matrix->stride;
    int work_size = 0;
    int h_info = 0;
    uint8_t error_code = MATH_QR_FAILED;

    float *d_work = NULL;
    float *d_tau = NULL;
    int *d_info = NULL;

    if (cusolverDnSgeqrf_bufferSize(solver_handle, m, n, matrix->matrix, lda, &work_size) != CUSOLVER_STATUS_SUCCESS)
        goto cleanup;
    if (work_size <= 0)
        goto cleanup;

    if (cudaMalloc(&d_work, sizeof(float) * (size_t)work_size) != cudaSuccess)
        goto cleanup;
    if (cudaMalloc(&d_tau, sizeof(float) * (size_t)n) != cudaSuccess)
        goto cleanup;
    if (cudaMalloc(&d_info, sizeof(int)) != cudaSuccess)
        goto cleanup;

    // 2. Compute QR Factorization (A = Q*R)
    if (cusolverDnSgeqrf(solver_handle, m, n, matrix->matrix, lda, d_tau, d_work, work_size, d_info) != CUSOLVER_STATUS_SUCCESS)
        goto cleanup;

    // 3. Extract Q from the Householder vectors
    if (cusolverDnSorgqr_bufferSize(solver_handle, m, n, n, matrix->matrix, lda, d_tau, &work_size) != CUSOLVER_STATUS_SUCCESS)
        goto cleanup;
    if (work_size <= 0)
        goto cleanup;

    cudaFree(d_work);
    d_work = NULL;
    if (cudaMalloc(&d_work, sizeof(float) * (size_t)work_size) != cudaSuccess)
        goto cleanup;

    if (cusolverDnSorgqr(solver_handle, m, n, n, matrix->matrix, lda, d_tau, d_work, work_size, d_info) != CUSOLVER_STATUS_SUCCESS)
        goto cleanup;

    if (cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
        goto cleanup;
    if (h_info != 0)
        goto cleanup;

    error_code = MATH_OPS_SUCCESS;

cleanup:
    if (d_work != NULL)
        cudaFree(d_work);
    if (d_tau != NULL)
        cudaFree(d_tau);
    if (d_info != NULL)
        cudaFree(d_info);

    return error_code;
}


