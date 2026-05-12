#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <math.h>
#include <string.h>
#include "../include/errors.h"
#include "../include/lin_alg.h"


matrix_t* lin_alg_create_matrix(const size_t m, const size_t n) {
    if (m == 0 || n == 0)
        return NULL;

    matrix_t *matrix_var = NULL;

    if ((matrix_var = (matrix_t*) malloc(sizeof(matrix_t))) == NULL)
        return NULL;
    
    matrix_var->stride = (n + 7) & ~7;
    size_t total_bytes = m * (matrix_var->stride) * sizeof(float);

    if (posix_memalign((void**) &matrix_var->matrix, 32, total_bytes) != 0)
        return NULL;

    memset(matrix_var->matrix, 0, total_bytes);

    matrix_var->m = m;
    matrix_var->n = n;

    return matrix_var;
}

int lin_alg_free_matrix(matrix_t **matrix) {
    if (matrix == NULL)
        return ERROR;

    if (*matrix) {
        if ((*matrix)->matrix)
            free((*matrix)->matrix);
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

    if (posix_memalign((void**) &vector->vector, 32, n * sizeof(aligned_float)) != 0)
        return NULL;

    vector->n = n;

    return vector;
}

int lin_alg_free_vector(vector_t ** vector) {
    if (vector == NULL)
        return ERROR;

    if (*vector) {
        if ((*vector)->vector)
            free((*vector)->vector);

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

    __m256 zero_reg = _mm256_setzero_ps();

    size_t i = 0;
    if (vector->n >= 8) {
        for (; i <= vector->n - 8; i += 8) 
            _mm256_store_ps(&vector->vector[i], zero_reg);
    }

    for (; i < vector->n; i++) {
        vector->vector[i] = 0.0f;
    }

    return SUCCESS;
}

/* Copy vec_2 into vec_1 */
int lin_alg_copy_vector(vector_t * vector_dest, const vector_t *vector_src) {
    if (vector_src == NULL || vector_dest == NULL)
        return ERROR;
    else if (vector_src->vector == NULL || vector_dest->vector == NULL ||
            vector_src->n == 0 || vector_dest->n == 0 || vector_src->n != vector_dest->n)
        return ERROR;

    __m256 temp_reg;

    size_t i = 0;

    if (vector_src->n >= 8) {
        for (; i <= vector_src->n - 8; i+= 8) { 
            temp_reg = _mm256_load_ps(&vector_src->vector[i]);
            _mm256_store_ps(&vector_dest->vector[i], temp_reg);
            
        }
    }

    for (; i < vector_src->n; i++)
        vector_dest->vector[i] = vector_src->vector[i];

    return SUCCESS;
}

/* Add vec_1 to vec_2, store result in vec_1 */
int lin_alg_add_vectors(vector_t *vector_dest, const vector_t *vector_src) {
    if (vector_dest == NULL || vector_src == NULL)
        return ERROR;
    else if (vector_src->vector == NULL || vector_dest->vector == NULL ||
            vector_src->n == 0 || vector_dest->n == 0 || vector_src->n != vector_dest->n)
        return ERROR;

    size_t i = 0;
    for (; i + 15 <= vector_src->n; i += 16) {
        __m256 a1 = _mm256_load_ps(&vector_dest->vector[i]);
        __m256 b1 = _mm256_load_ps(&vector_src->vector[i]);
        __m256 a2 = _mm256_load_ps(&vector_dest->vector[i + 8]);
        __m256 b2 = _mm256_load_ps(&vector_src->vector[i + 8]);

        __m256 r1 = _mm256_add_ps(a1, b1);
        __m256 r2 = _mm256_add_ps(a2, b2);
        _mm256_store_ps(&vector_dest->vector[i], r1);
        _mm256_store_ps(&vector_dest->vector[i + 8], r2);
    }

    if (i + 7 < vector_src->n) {
        __m256 a1 = _mm256_load_ps(&vector_dest->vector[i]);
        __m256 b1 = _mm256_load_ps(&vector_src->vector[i]);

        __m256 r = _mm256_add_ps(a1, b1);
        _mm256_store_ps(&vector_src->vector[i], r);
        i += 8;
    }

    for (; i < vector_src->n; i++)
        vector_dest->vector[i] += vector_src->vector[i];

    return SUCCESS;
}

/* Substract vec_2 from vec_1, store result in vec_1 */
int lin_alg_sub_vectors(vector_t *vector_dest, const vector_t *vector_src) {
    if (vector_dest == NULL || vector_src == NULL)
        return ERROR;
    else if (vector_src->vector == NULL || vector_dest->vector == NULL ||
            vector_src->n == 0 || vector_dest->n == 0 || vector_src->n != vector_dest->n)
        return ERROR;

    size_t i = 0;
    for (; i + 15 < vector_src->n; i += 16) {
        __m256 a1 = _mm256_load_ps(&vector_dest->vector[i]);
        __m256 b1 = _mm256_load_ps(&vector_src->vector[i]);
        __m256 a2 = _mm256_load_ps(&vector_dest->vector[i + 8]);
        __m256 b2 = _mm256_load_ps(&vector_src->vector[i + 8]);

        __m256 r1 = _mm256_sub_ps(a1, b1);
        __m256 r2 = _mm256_sub_ps(a2, b2);
        _mm256_store_ps(&vector_dest->vector[i], r1);
        _mm256_store_ps(&vector_dest->vector[i + 8], r2);
    }

    if (i + 7 < vector_src->n) {
        __m256 a1 = _mm256_load_ps(&vector_dest->vector[i]);
        __m256 b1 = _mm256_load_ps(&vector_src->vector[i]);

        __m256 r = _mm256_sub_ps(a1, b1);
        _mm256_store_ps(&vector_dest->vector[i], r);
        i += 8;
    }

    for (; i < vector_src->n; i++)
        vector_dest->vector[i] -= vector_src->vector[i];


    return SUCCESS;
}

/* Scale vec with constant α, store result in vec */
int lin_alg_scale_vector(vector_t *vector, const float α) {
    if (vector == NULL)
        return ERROR;
    else if (vector->vector == NULL || vector->n == 0)
        return ERROR;

    __m256 scale_reg = _mm256_set1_ps(α); 
    size_t i = 0;
    
    for (; i + 15 < vector->n; i += 16) {
        __m256 a1 = _mm256_load_ps(&vector->vector[i]);
        __m256 a2 = _mm256_load_ps(&vector->vector[i + 8]);

        __m256 r1 = _mm256_mul_ps(a1, scale_reg);
        __m256 r2 = _mm256_mul_ps(a2, scale_reg);
        _mm256_store_ps(&vector->vector[i], r1);
        _mm256_store_ps(&vector->vector[i + 8], r2);
    }

    if (i + 7 < vector->n) {
        __m256 a1 = _mm256_load_ps(&vector->vector[i]);

        __m256 r = _mm256_mul_ps(a1, scale_reg);
        _mm256_store_ps(&vector->vector[i], r);
        i += 8;
    }

    for (; i < vector->n; i++)
        vector->vector[i] *= α;

    return SUCCESS;
}

/* Dot product of vec_1 and vec_2, return result */
int lin_alg_dot_productv(const vector_t *vec_1, const vector_t* vec_2, aligned_float *result) {
    if (vec_1 == NULL || vec_2 == NULL || result == NULL)
        return ERROR;
    else if (vec_1->vector == NULL || vec_2->vector == NULL || 
            vec_1->n != vec_2->n)
        return ERROR;

    size_t i = 0;
    aligned_float _result = 0.0f;

    if (vec_1->n >= 8) {
        __m256 accum_reg = _mm256_setzero_ps();

        for (; i + 7 < vec_1->n; i += 8) {
            __m256 reg_vec_1 = _mm256_load_ps(&vec_1->vector[i]);
            __m256 reg_vec_2 = _mm256_load_ps(&vec_2->vector[i]);

            accum_reg = _mm256_fmadd_ps(reg_vec_1, reg_vec_2, accum_reg);
        }

        __m128 low128 = _mm256_castps256_ps128(accum_reg);
        __m128 high128 = _mm256_extractf128_ps(accum_reg, 1);
        __m128 sum128 = _mm_add_ps(low128, high128);

        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);

        _result = _mm_cvtss_f32(sum128);
    }

    for (;i < vec_1->n; i++)
        _result += (vec_1->vector[i] * vec_2->vector[i]);

    *result = _result;

    return SUCCESS;
}

/* Dot product of mtx and vec, store result in res */
int lin_alg_dot_productmv(const matrix_t *matrix, const vector_t *vector, vector_t *result) {
    if (matrix == NULL || vector == NULL || result == NULL)
        return ERROR;
    else if (matrix->matrix == NULL || vector->vector == NULL || 
            matrix->n != vector->n)
        return ERROR;

    vector_t row_ptr;
    for (size_t row = 0; row < matrix->m; row++) {
        row_ptr.vector = &matrix->matrix[row * matrix->stride];
        row_ptr.n = matrix->n;

        if (lin_alg_dot_productv(&row_ptr, vector, &result->vector[row]) != SUCCESS)
            return ERROR;
    }

    return SUCCESS;
}

/* Print vector to stdout */
int lin_alg_display_vector(const vector_t * vector) {
    if (vector == NULL)
        return ERROR;
    else if (vector->vector == NULL || vector->n == 0)
        return ERROR;

    fprintf(stdout, "{");
    for (size_t i = 0; i < vector->n; i++) {
        fprintf(stdout, "%.2f", vector->vector[i]);
        if (i != vector->n - 1)
            fprintf(stdout, ",");
    }
    fprintf(stdout, "}\n");

    return SUCCESS;
}

matrix_t* lin_alg_transpose_matrix(const matrix_t *matrix) {
    if (matrix == NULL)
        return NULL;
    else if (matrix->m == 0 || matrix->n == 0 ||
            matrix->matrix == NULL)
        return NULL;

    // Create the new matrix: Rows = original 'n', Cols = original 'm'
    matrix_t *matrix_ = lin_alg_create_matrix(matrix->n, matrix->m);
    if (matrix_ == NULL)
        return NULL;

    // size_t s2 = *stride_2; // The stride of the new transposed matrix

    for (size_t r = 0; r < matrix->m; r++) {
        size_t c = 0;

        // 1. SIMD Load: Process columns 8 at a time
        for (; c + 7 < matrix->n; c += 8) {

            // Load 8 floats from Row 'r' of the original matrix
            __m256 row_chunk = _mm256_load_ps(&matrix->matrix[r * matrix->stride + c]);

            // 2. Manual Scatter: Place each float into its new Row (original Column)
            // mtx_[col][row] = mtx[row][col]
            // We use a temporary array or direct extraction to avoid the AVX2 scatter limit
            float temp[8];
            _mm256_storeu_ps(temp, row_chunk);

            matrix_->matrix[(c + 0) * matrix_->stride + r] = temp[0];
            matrix_->matrix[(c + 1) * matrix_->stride + r] = temp[1];
            matrix_->matrix[(c + 2) * matrix_->stride + r] = temp[2];
            matrix_->matrix[(c + 3) * matrix_->stride + r] = temp[3];
            matrix_->matrix[(c + 4) * matrix_->stride + r] = temp[4];
            matrix_->matrix[(c + 5) * matrix_->stride + r] = temp[5];
            matrix_->matrix[(c + 6) * matrix_->stride + r] = temp[6];
            matrix_->matrix[(c + 7) * matrix_->stride + r] = temp[7];
        }

        // 3. Scalar Tail: Handle remaining columns if n is not a multiple of 8
        for (; c < matrix->n; c++) {
            matrix_->matrix[c * matrix_->stride + r] = matrix->matrix[r * matrix->stride + c];
        }
    }

    return matrix_;
}

aligned_float lin_alg_l2(const vector_t *vec) {
    if (vec == NULL)
        return -1.0f;
    else if (vec->vector == NULL || vec->n == 0)
        return -1.0f;

    aligned_float l2_norm = 0.0f;

    if (lin_alg_dot_productv(vec, vec, &l2_norm) != SUCCESS)
        return -1.0f;

    l2_norm = sqrtf(l2_norm);
    return l2_norm;
}

/* Involve vectorization for parallel processing */
uint8_t lin_alg_l2_normalize(vector_t *vec) {
    if (vec == NULL)
        return MATH_OPS_NULL;
    else if (vec->vector == NULL || vec->n == 0)
        return MATH_OPS_EMPTY; 

    aligned_float l2_norm = lin_alg_l2(vec);
    if (l2_norm < 0.0f)
        return MATH_L2_FAILED;

    if (l2_norm < 1e-12f) return MATH_L2_FAILED;

    aligned_float scaler = 1/l2_norm;
    if (lin_alg_scale_vector(vec, scaler) != SUCCESS)
        return MATH_L2_FAILED;

    return MATH_OPS_SUCCESS;
}

// Updated state to use 32-bit lanes (8 lanes total)
struct _xorshift_state {
    __m256i s; 
};

__m256i xorshift32_avx2(struct _xorshift_state* state) {
    __m256i x = state->s;
    x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 13));
    x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 17));
    x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 5));
    state->s = x;
    return x;
}

matrix_t* lin_alg_normal_rand_matrix(const uint16_t n) {
    if (n == 0) return NULL;

    matrix_t *matrix = lin_alg_create_matrix(n, n);
    if (matrix == NULL) return NULL;

    struct _xorshift_state state;
    // Initialize 8 distinct seeds for the 8 lanes
    state.s = _mm256_setr_epi32(rand(), rand(), rand(), rand(), 
                                rand(), rand(), rand(), rand());

    __m256 two_pi = _mm256_set1_ps(2.0f * (float)PI);
    __m256 minus_two = _mm256_set1_ps(-2.0f);
    __m256 eps = _mm256_set1_ps(1e-10f); // Small epsilon to avoid log(0)
    
    // Multiplier for 32-bit unsigned range (approx 1.0 / 4.29e9)
    // We use a slightly smaller value to keep results in [0, 1)
    __m256 inv_max_32 = _mm256_set1_ps(1.0f / 4294967296.0f);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j + 7 < n; j += 8) {
            // 1. Generate 8x32-bit random integers
            __m256i raw_x1 = xorshift32_avx2(&state);
            __m256i raw_x2 = xorshift32_avx2(&state);

            // 2. Convert to Float [0, 1)
            // Use _mm256_and_si256 with a mask to treat as positive ints
            __m256i mask = _mm256_set1_epi32(0x7FFFFFFF); 
            __m256 x1 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(raw_x1, mask)), inv_max_32), eps);
            __m256 x2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(raw_x2, mask)), inv_max_32);

            // 3. Box-Muller (Requires SVML)
            __m256 log_x1 = _mm256_log_ps(x1);    // Intel SVML
            __m256 mag = _mm256_sqrt_ps(_mm256_mul_ps(minus_two, log_x1));
            __m256 angle = _mm256_mul_ps(two_pi, x2);
            
            __m256 cos_angle = _mm256_cos_ps(angle); // Intel SVML
            __m256 result = _mm256_mul_ps(mag, cos_angle);

            // 4. Store
            _mm256_store_ps(&matrix->matrix[i * matrix->stride + j], result);
        }

        for (size_t j_tail = (n & ~7); j_tail < n; j_tail++) {
            // Simple scalar Box-Muller or random fill
            aligned_float u1 = (float)rand() / (float)RAND_MAX + 1e-10f;
            aligned_float u2 = (float)rand() / (float)RAND_MAX;
            aligned_float mag = sqrtf(-2.0f * logf(u1));
            matrix->matrix[i * matrix->stride + j_tail] = mag * cosf(2.0f * (float)PI * u2);
        }
    }
    return matrix;
}

/* Modified Gram-Schmidt (MGS) algorithm to do so */
uint8_t lin_alg_qr_decompose(matrix_t *matrix) {
    if (matrix == NULL)
        return MATH_OPS_NULL;
    else if (matrix->n == 0)
        return MATH_OPS_EMPTY;

    matrix_t *t_matrix = NULL, *temp_matrix = NULL;
    vector_t *temp_vec = NULL, pivot_vec, sliding_vec; 
    aligned_float dot = 0.0f;
    
    t_matrix = lin_alg_transpose_matrix(matrix);
    if (t_matrix == NULL)
        return MATH_OPS_NULL;

    temp_vec = lin_alg_create_vector(matrix->n);
    if (temp_vec == NULL) {
        lin_alg_free_matrix(&t_matrix);
        return MATH_OPS_NULL;
    }

    pivot_vec.n = t_matrix->n;
    sliding_vec.n = t_matrix->n;

    uint16_t error_code = MATH_OPS_SUCCESS;
    
    for (int k = 0; k < matrix->n; k++) {
        pivot_vec.vector = &t_matrix->matrix[k * t_matrix->stride];
        if ((error_code = lin_alg_l2_normalize(&pivot_vec)) != MATH_OPS_SUCCESS) 
            goto cleanup;

        for (int i = k + 1; i < matrix->n; i++) {
            sliding_vec.vector = &t_matrix->matrix[i * t_matrix->stride];
            if (lin_alg_dot_productv(&pivot_vec, &sliding_vec, &dot) != 0) {
                error_code = MATH_QR_FAILED;
                goto cleanup;
            }

            if (lin_alg_copy_vector(temp_vec, &pivot_vec) != 0) {
                error_code = MATH_QR_FAILED;
                goto cleanup;
            }

            if (lin_alg_scale_vector(temp_vec, dot) != 0) {
                error_code = MATH_QR_FAILED;
                goto cleanup;
            }

            if (lin_alg_sub_vectors(&sliding_vec, temp_vec) != 0) {
                error_code = MATH_QR_FAILED;
                goto cleanup;
            }
        }

    }

    /* Check diagonal values if sign was flipped */
    for (int i = 0; i < matrix->n; i++) {
        if (t_matrix->matrix[i * t_matrix->stride + i] < 0.0f) {
            pivot_vec.vector = &t_matrix->matrix[i * t_matrix->stride];
            if (lin_alg_scale_vector(&pivot_vec, -1.0f) != 0) {
                error_code = MATH_QR_FAILED;
                goto cleanup;
            }
        }
    }


    temp_matrix = lin_alg_transpose_matrix(t_matrix);
    lin_alg_free_matrix(&t_matrix);

    for (int i = 0; i < matrix->n; i++) {
        pivot_vec.vector = &matrix->matrix[i * matrix->stride];
        sliding_vec.vector = &temp_matrix->matrix[i * temp_matrix->stride];
        if (lin_alg_copy_vector(&pivot_vec, &sliding_vec) != 0) {
            error_code = MATH_QR_FAILED;
            goto cleanup;
        }
    }

cleanup:
    if (temp_vec != NULL) lin_alg_free_vector(&temp_vec);
    if (temp_matrix != NULL) lin_alg_free_matrix(&temp_matrix);

    if (error_code != MATH_OPS_SUCCESS) {
        return error_code;
    } else
        return MATH_OPS_SUCCESS;
}
