#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/errors.h"
#include "../include/lin_alg.h"

float** create_mtx(const size_t m, const size_t n) {
    if (m == 0 || n == 0)
        return NULL;

    float **mtx = (float**) calloc(m, sizeof(float*));
    if (mtx == NULL)
        return NULL;

    for (size_t i = 0; i < m; i++) {
        mtx[i] = (float*) calloc(n, sizeof(float));
        if (mtx[i] == NULL) {
            for (size_t j = 0; j < i; j++)
                free(mtx[j]);
            free(mtx);
            return NULL;
        }
    }

    return mtx;
}

int free_mtx(float **mtx, const size_t m) {
    if (mtx == NULL || m == 0)
        return ERROR;

    for (size_t i = 0; i < m; i++) {
        if (mtx[i] != NULL)
            free(mtx[i]);
    }
    free(mtx);

    return SUCCESS;
}

float* create_vec(const size_t n) {
    if (n == 0)
        return NULL;

    float *vec = (float*) calloc(n, sizeof(float));
    if (vec == NULL)
        return NULL;

    return vec;
}

int free_vec(float *const vec) {
    if (vec == NULL)
        return ERROR;

    free(vec);
    return SUCCESS;
}

/* Zero vector / clear a vector */
int zero_vec(float *vec, const size_t n) {
    if (vec == NULL || n == 0)
        return ERROR;

    for (size_t i = 0; i < n; i++)
        vec[i] = 0.0f;

    return SUCCESS;
}

/* Copy vec_2 into vec_1 */
int copy_vec(float *vec_1, const float *vec_2, const size_t n) {
    if (vec_1 == NULL || vec_2 == NULL || n == 0)
        return ERROR;

    for (size_t i = 0; i < n; i++)
        vec_1[i] = vec_2[i];

    return SUCCESS;
}

/* Add vec_1 to vec_2, store result in vec_1 */
int add_vec(float *vec_1, const float *vec_2, const size_t n) {
    if (vec_1 == NULL || vec_2 == NULL || n == 0)
        return ERROR;

    for (size_t i = 0; i < n; i++)
        vec_1[i] += vec_2[i];

    return SUCCESS;
}

/* Substract vec_2 from vec_1, store result in vec_1 */
int sub_vec(float *vec_1, const float *vec_2, const size_t n) {
    if (vec_1 == NULL || vec_2 == NULL || n == 0)
        return ERROR;

    for (size_t i = 0; i < n; i++) 
        vec_1[i] -= vec_2[i];

    return SUCCESS;
}

/* Scale vec with constant α, store result in vec */
int scale_vec(float *vec, const float α, const size_t n) {
    if (vec == NULL || n == 0)
        return ERROR;

    for (size_t i = 0; i < n; i++)
        vec[i] *= α;

    return SUCCESS;
}

/* Dot product of vec_1 and vec_2, return result */
int dot_product_vec(const float *vec_1, const float *vec_2, const size_t n, float *res) {
    if (vec_1 == NULL || vec_2 == NULL || n == 0 || res == NULL)
        return ERROR;

    *res = 0.0f;

    for (size_t i = 0; i < n; i++)
        (*res) += (vec_1[i] * vec_2[i]);

    return SUCCESS;
}

/* Dot product of mtx and vec, store result in res */
int dot_product_mtx_vec(const float **mtx, const float *vec, const size_t m, const size_t n_mtx, const size_t n_vec,
        float *res) {
    if (mtx == NULL || vec == NULL || res == NULL)
        return ERROR;
    if (n_mtx != n_vec)
        return ERROR;

    for (size_t i = 0; i < m; i++) {
        if (dot_product_vec(mtx[i], vec, n_mtx, &res[i]) != SUCCESS)
            return ERROR;
    }

    return SUCCESS;
}

/* Print vector to stdout */
int display_vec(const float *vec_1, const size_t n) {
    if (vec_1 == NULL || n == 0)
        return ERROR;

    fprintf(stdout, "{");
    for (size_t i = 0; i < n; i++) {
        fprintf(stdout, "%.2f", vec_1[i]);
        if (i != n-1)
            fprintf(stdout, ",");
    }
    fprintf(stdout, "}\n");

    return SUCCESS;
}

/* Transpose a matrix mtx */
float** transpose_mtx(const float **mtx, const size_t m, const size_t n) {
    if (mtx == NULL || m == 0 || n == 0)
        return NULL;

    float **mtx_ = create_mtx(n, m);
    if (mtx_ == NULL)
        return NULL;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            mtx_[i][j] = mtx[j][i];
        }
    }

    return mtx_;
}

/* LU decomposition of mtx */
int LU_decomposition_mtx(const float **mtx, float **L, float **U, const size_t m, const size_t n) {
    if (mtx == NULL || L == NULL || U == NULL)
        return ERROR;
    if (m == 0 || n == 0)
        return ERROR;

    /* Initialize L as identity matrix */
    for (size_t i = 0; i < m; i++) {
        if (i < n)
            L[i][i] = 1.0f;
        else
            break;
    }

    /* Copy mtx into U */
    for (size_t i = 0; i < m; i++) {
        if (copy_vec(U[i], mtx[i], n) != SUCCESS)
            return ERROR;
    }

    size_t min_dim = (m < n) ? m : n;
    for (size_t i = 0; i < min_dim; i++) {
        if (fabsf(U[i][i]) < EPSILON)
            return ERROR;
        
        for (size_t j = i + 1; j < m; j++) {
            float divisor = U[j][i] / U[i][i];
            L[j][i] = divisor;
            
            for (size_t k = i; k < n; k++) {
                U[j][k] -= divisor * U[i][k];
            }
        }
    }

    return SUCCESS;
}

/* Get determinant of a matrix */
int determinant_mtx(const float **mtx, const size_t m, const size_t n, float *result) {
    if (mtx == NULL || result == NULL)
        return ERROR;
    if (m == 0 || n == 0)
        return ERROR;

    if (m == 2 && n == 2) {
        *result = (mtx[0][0] * mtx[1][1]) - (mtx[0][1] * mtx[1][0]);
        return SUCCESS;
    } else if (m == 1 && n == 1) {
        *result = mtx[0][0];
        return SUCCESS;
    }

    int err = SUCCESS;
    float **L = create_mtx(m, n), **U = create_mtx(m, n);
    if (L == NULL)
        return ERROR;
    else if (U == NULL) {
        free_mtx(L, m);
        return ERROR;
    }

    err = LU_decomposition_mtx(mtx, L, U, m, n);
    if (err != SUCCESS) {
        free_mtx(L, m);
        free_mtx(U, m);
        return ERROR;
    }

    float product = 1.0f;
    for (size_t i = 0; i < m; i++) {
        if (i < n)
            product *= U[i][i];
        else
            break;
    }

    *result = product;

    free_mtx(U, m);
    free_mtx(L, m);
    return SUCCESS;
}

/* Check if a matrix mtx is invertible */
int is_invertible(const float determinant) {
    return (fabsf(determinant) > EPSILON);
}

float** inverse_mtx(const float **mtx, const size_t m, const size_t n) {
    if (mtx == NULL)
        return NULL;
    if (m == 0 || n == 0 || m != n)  // Matrix must be square
        return NULL;
    
    /* Check if matrix is invertible */
    float det = 0.0f;
    if (determinant_mtx(mtx, m, n, &det) != SUCCESS || !is_invertible(det))
        return NULL;
    
    /* Create augmented matrix [A | I] */
    float **aug = create_mtx(n, 2 * n);
    if (aug == NULL)
        return NULL;
    
    /* Copy matrix A and identity matrix I into augmented matrix */
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            aug[i][j] = mtx[i][j];        /* Copy A */
            aug[i][j + n] = (i == j) ? 1.0f : 0.0f;  /* Identity matrix */
        }
    }
    
    /* Gauss-Jordan elimination */
    for (size_t i = 0; i < n; i++) {
        /* Find pivot */
        size_t pivot_row = i;
        for (size_t k = i + 1; k < n; k++) {
            if (fabsf(aug[k][i]) > fabsf(aug[pivot_row][i]))
                pivot_row = k;
        }
        
        /* Swap rows if needed */
        if (pivot_row != i) {
            float *temp = aug[i];
            aug[i] = aug[pivot_row];
            aug[pivot_row] = temp;
        }
        
        /* Check for zero pivot */
        if (fabsf(aug[i][i]) < EPSILON) {
            free_mtx(aug, n);
            return NULL;
        }
        
        /* Scale pivot row */
        float pivot = aug[i][i];
        for (size_t j = 0; j < 2 * n; j++) {
            aug[i][j] /= pivot;
        }
        
        /* Eliminate column */
        for (size_t k = 0; k < n; k++) {
            if (k != i) {
                float factor = aug[k][i];
                for (size_t j = 0; j < 2 * n; j++) {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
    }
    
    /* Extract inverse from augmented matrix */
    float **inv = create_mtx(n, n);
    if (inv == NULL) {
        free_mtx(aug, n);
        return NULL;
    }
    
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            inv[i][j] = aug[i][j + n];
        }
    }
    
    free_mtx(aug, n);
    return inv;
}

/* Return a column copy of index cidx from matrix mtx */
float* get_column_mtx(const float **mtx, const size_t m, const size_t n, const size_t cidx) {
    if (mtx == NULL)
        return NULL;
    else if (cidx >= n)
        return NULL;
    else if (m == 0 || n == 0)
        return NULL;

    float *column = NULL;
    if ((column = create_vec(m)) == NULL)
        return NULL;

    for (size_t i = 0; i < m; i++)
        column[i] = mtx[i][cidx];

    return column;
}
