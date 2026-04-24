#ifndef LIN_ALG_H
#define LIN_ALG_H

#include <stddef.h>

#define EPSILON 1e-10f

/* The linear algebra module provides an API for vector and matrix operations */

/* 
 * Description: Create a dynamically allocated matrix initialized to zero
 * Input: m (number of rows), n (number of columns)
 * Return: Pointer to allocated matrix, or NULL on failure
 */
float** create_mtx(const size_t m, const size_t n);

/* 
 * Description: Free the memory allocated for a matrix
 * Input: mtx (pointer to matrix), m (number of rows)
 * Return: SUCCESS on success, ERROR on failure
 */
int free_mtx(float **mtx, const size_t m);

/* 
 * Description: Create a dynamically allocated vector initialized to zero
 * Input: n (number of elements)
 * Return: Pointer to allocated vector, or NULL on failure
 */
float* create_vec(const size_t n);

/* 
 * Description: Free the memory allocated for a vector
 * Input: vec (pointer to vector)
 * Return: SUCCESS on success, ERROR on failure
 */
int free_vec(float * const vec);

/* 
 * Description: Set all elements of a vector to zero
 * Input: vec (pointer to vector), n (number of elements)
 * Return: SUCCESS on success, ERROR on failure
 */
int zero_vec(float *vec, const size_t n);

/* 
 * Description: Copy elements from vec_2 into vec_1
 * Input: vec_1 (destination vector), vec_2 (source vector), n (number of elements)
 * Return: SUCCESS on success, ERROR on failure
 */
int copy_vec(float *vec_1, const float *vec_2, const size_t n);

/* 
 * Description: Add vec_2 to vec_1 element-wise, store result in vec_1
 * Input: vec_1 (first vector, also destination), vec_2 (second vector), n (number of elements)
 * Return: SUCCESS on success, ERROR on failure
 */
int add_vec(float *vec_1, const float *vec_2, const size_t n);

/* 
 * Description: Subtract vec_2 from vec_1 element-wise, store result in vec_1
 * Input: vec_1 (first vector, also destination), vec_2 (second vector), n (number of elements)
 * Return: SUCCESS on success, ERROR on failure
 */
int sub_vec(float *vec_1, const float *vec_2, const size_t n);

/* 
 * Description: Multiply each element of vec by scalar α, store result in vec
 * Input: vec (vector to scale), α (scaling factor), n (number of elements)
 * Return: SUCCESS on success, ERROR on failure
 */
int scale_vec(float *vec, const float α, const size_t n);

/* 
 * Description: Compute the dot product of vec_1 and vec_2
 * Input: vec_1 (first vector), vec_2 (second vector), n (number of elements), res (pointer to store result)
 * Return: SUCCESS on success, ERROR on failure
 */
int dot_product_vec(const float *vec_1, const float *vec_2, const size_t n, float *res);

/* 
 * Description: Compute matrix-vector product mtx × vec, store result in res
 * Input: mtx (m×n matrix), vec (vector of length n), m (number of rows), n_mtx (number of columns in mtx), n_vec (length of vec), res (result vector)
 * Return: SUCCESS on success, ERROR on failure
 */
int dot_product_mtx_vec(const float **mtx, const float *vec, const size_t m, const size_t n_mtx, const size_t n_vec,
        float *res);

/* 
 * Description: Print vector elements to stdout in format {e1,e2,...,en}
 * Input: vec_1 (vector to display), n (number of elements)
 * Return: SUCCESS on success, ERROR on failure
 */
int display_vec(const float *vec_1, const size_t n);

/* 
 * Description: Compute the transpose of matrix mtx
 * Input: mtx (m×n matrix), m (number of rows), n (number of columns)
 * Return: Pointer to transposed n×m matrix, or NULL on failure
 */
float** transpose_mtx(const float **mtx, const size_t m, const size_t n);

/* 
 * Description: Perform LU decomposition on matrix mtx, storing L and U factors
 * Input: mtx (input matrix), L (lower triangular matrix), U (upper triangular matrix), m (number of rows), n (number of columns)
 * Return: SUCCESS on success, ERROR on failure
 */
int LU_decomposition_mtx(const float **mtx, float **L, float **U, const size_t m, const size_t n);

/* 
 * Description: Calculate the determinant of a square matrix using LU decomposition
 * Input: mtx (square matrix), m (number of rows), n (number of columns), result (pointer to store determinant)
 * Return: SUCCESS on success, ERROR on failure
 */
int determinant_mtx(const float **mtx, const size_t m, const size_t n, float *result);

/* 
 * Description: Check if a matrix is invertible based on its determinant
 * Input: determinant (determinant value)
 * Return: 1 if invertible (|det| > EPSILON), 0 otherwise
 */
int is_invertible(const float determinant);

/* 
 * Description: Compute the inverse of a square matrix using Gauss-Jordan elimination
 * Input: mtx (square matrix to invert), m (number of rows), n (number of columns)
 * Return: Pointer to inverse matrix, or NULL if not invertible or on failure
 */
float** inverse_mtx(const float **mtx, const size_t m, const size_t n);

/* 
 * Description: Extract and return a copy of column cidx from matrix mtx
 * Input: mtx (matrix), m (number of rows), n (number of columns), cidx (column index)
 * Return: Pointer to column vector, or NULL on failure
 */
float* get_column_mtx(const float **mtx, const size_t m, const size_t n, const size_t cidx);

#endif
