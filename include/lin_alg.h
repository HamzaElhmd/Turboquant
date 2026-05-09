#ifndef LIN_ALG_H
#define LIN_ALG_H

#include <stddef.h>
#include <stdint.h> 

/* TODO: Initialization the cuda pointer mode to device for optimization */

#define EPSILON 1e-10f
#define PI 3.14159265358979323846264338327950288f

#ifdef __cplusplus
extern "C" {
#endif


/* Usage of a row order matrix
 * The stride takes into account the number of elements
 * until the next row + number of 32-bit paddings for 
 * 32 byte memory alignment
 */
typedef struct {
        size_t m;
        size_t n;
        size_t stride;
        float *matrix;
        uint8_t transpose_flag;
} matrix_t;

typedef struct {
    size_t n;
    float *vector;
} vector_t;

/* The linear algebra module provides an API for vector and matrix operations */

/*
 * Description: Initialize shared CUDA math handles used by linear algebra ops
 * Return: SUCCESS on success, ERROR on failure
 */
int lin_alg_runtime_init(void);

/*
 * Description: Destroy shared CUDA math handles used by linear algebra ops
 * Return: SUCCESS on success, ERROR on failure
 */
int lin_alg_runtime_shutdown(void);

/*
 * Description: Set CUDA stream for cuBLAS/cuSOLVER calls
 * Input: stream_handle (cudaStream_t cast to void*)
 * Return: SUCCESS on success, ERROR on failure
 */
int lin_alg_set_stream(void *stream_handle);

/* 
 * Description: Create a dynamically allocated matrix initialized to zero
 * Input: m (number of rows), n (number of columns)
 * Return: Pointer to allocated matrix, or NULL on failure
 */
matrix_t* lin_alg_create_matrix(const size_t m, const size_t n);

/* 
 * Description: Free the memory allocated for a matrix
 * Input: mtx (pointer to matrix), m (number of rows)
 * Return: SUCCESS on success, ERROR on failure
 */
int lin_alg_free_matrix(matrix_t **matrix);

/* 
 * Description: Create a dynamically allocated vector initialized to zero
 * Input: n (number of elements)
 * Return: Pointer to allocated vector, or NULL on failure
 */
vector_t* lin_alg_create_vector(const size_t n);

/* 
 * Description: Free the memory allocated for a vector
 * Input: vec (pointer to vector)
 * Return: SUCCESS on success, ERROR on failure
 */
int lin_alg_free_vector(vector_t ** vector);

/* 
 * Description: Set all elements of a vector to zero
 * Input: vec (pointer to vector), n (number of elements)
 * Return: SUCCESS on success, ERROR on failure
 */
int lin_alg_zero_vector(vector_t *vec);

/* 
 * Description: Copy elements from vec_2 into vec_1
 * Input: vec_1 (destination vector), vec_2 (source vector)
 * Return: SUCCESS on success, ERROR on failure
 */
int lin_alg_copy_vector(vector_t * vector_dest, const vector_t *vector_src);


/* 
 * Description: Copy elements from matrix_src into matrix_dest
 * Input: matrix_dest (destination vector), matrix_src (source vector)
 * Return: SUCCESS on success, ERROR on failure
 */
int lin_alg_copy_matrix(matrix_t *matrix_dest, const matrix_t *matrix_src);


/* 
 * Description: Add vec_2 to vec_1 element-wise, store result in vec_1
 * Input: vec_1 (first vector, also destination), vec_2 (second vector), n (number of elements)
 * Return: SUCCESS on success, ERROR on failure
 */
int lin_alg_add_vectors(vector_t *vector_dest, const vector_t *vector_src);

/* 
 * Description: Subtract vec_2 from vec_1 element-wise, store result in vec_1
 * Input: vec_1 (first vector, also destination), vec_2 (second vector), n (number of elements)
 * Return: SUCCESS on success, ERROR on failure
 */
int lin_alg_sub_vectors(vector_t *vector_dest, const vector_t *vector_src);

/* 
 * Description: Multiply each element of vec by scalar α, store result in vec
 * Input: vec (vector to scale), α (scaling factor), n (number of elements)
 * Return: SUCCESS on success, ERROR on failure
 */
int lin_alg_scale_vector(vector_t *vector, const float α);

/* 
 * Description: Compute the dot product of vec_1 and vec_2
 * Input: vec_1 (first vector), vec_2 (second vector), n (number of elements), res (pointer to store result)
 * Return: SUCCESS on success, ERROR on failure
 */
int lin_alg_dot_productv(const vector_t *vec_1, const vector_t* vec_2, float *result);

/* 
 * Description: Compute matrix-vector product mtx × vec, store result in res
 * Input: mtx (m×n matrix), vec (vector of length n), m (number of rows), n_mtx (number of columns in mtx), n_vec (length of vec), res (result vector)
 * Return: SUCCESS on success, ERROR on failure
 */
int lin_alg_dot_productmv(const matrix_t *matrix, const vector_t *vector, vector_t *result);

/* 
 * Description: Compute the transpose of matrix mtx
 * Input: mtx (m×n matrix), m (number of rows), n (number of columns)
 * Return: Pointer to transposed n×m matrix, or NULL on failure
 */
matrix_t* lin_alg_transpose_matrix(matrix_t *matrix);

/* Computing the l2 norm of a vector */
float lin_alg_l2(const vector_t *vec);

/* TODO: Extend the input from uint16_t to uint32_t */
/* Calculate the L2 norm of input vectors to ensure length is 1,
 * projecting the vector on the hypersphere S^d-1 */
uint8_t lin_alg_l2_normalize(vector_t *vec);

/* Generate a square nxn matrix where each row is random normal distribution values */
matrix_t* lin_alg_normal_rand_matrix(matrix_t *mtx);

/* Apply QR decomposition a square matrix to ensure orthogonality and uniformity */
uint8_t lin_alg_qr_decompose(matrix_t *matrix);

#ifdef __cplusplus
}
#endif

#endif
