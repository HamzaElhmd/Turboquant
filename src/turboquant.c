#include "../include/turboquant.h"
#include "../include/errors.h"
#include "../include/config.h"
#include "../include/lin_alg.h"
#include <stdio.h>
#include <immintrin.h>
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

turbo_quantizer *mse_quantizer = NULL;
uint8_t is_init = 0;

void turboquant_quantizer_destroy(turbo_quantizer **quantizer) {
    if (quantizer) {
        if ((*quantizer)) {
            if ((*quantizer)->book) 
                codebook_destroy(&(*quantizer)->book);
            if((*quantizer)->Π) 
                lin_alg_free_matrix(&(*quantizer)->Π);
            if ((*quantizer)->t_Π)
                lin_alg_free_matrix(&(*quantizer)->t_Π);
            if ((*quantizer)->S)
                lin_alg_free_matrix(&(*quantizer)->S);
            if ((*quantizer)->t_S)
                lin_alg_free_matrix(&(*quantizer)->t_S);

            free((*quantizer));
            *quantizer = NULL;
        }
    }
}

turbo_quantizer* turboquant_quantizer_init(const size_t dims, const uint8_t bit_width) {
   
    uint8_t error_code = QUANT_SUCCESS;
    
    /* Initialize the Haar Matrix */
    turbo_quantizer *quantizer = (turbo_quantizer*) malloc(sizeof(turbo_quantizer));
    if (quantizer == NULL) 
        return NULL;

    quantizer->book = NULL;
    quantizer->t_Π = NULL;
    quantizer->Π = NULL;
    quantizer->S = NULL;
    quantizer->t_S = NULL;

    quantizer->S = lin_alg_normal_rand_matrix(dims);
    if (quantizer->S == NULL) {
        error_code = QUANT_INIT_FAILED;
        goto cleanup;
    }

    quantizer->Π = lin_alg_create_matrix(dims, dims);
    if (quantizer->Π == NULL) {
        error_code = QUANT_INIT_FAILED;
        goto cleanup;
    }

    vector_t src_vec, dest_vec;
    src_vec.n = dims, dest_vec.n = dims;
    for (int i = 0; i < dims; i++) {
        src_vec.vector = &quantizer->S->matrix[i * quantizer->S->stride];
        dest_vec.vector = &quantizer->Π->matrix[i * quantizer->Π->stride];
        if (lin_alg_copy_vector(&dest_vec, &src_vec) != SUCCESS) {
            error_code = QUANT_INIT_FAILED;
            goto cleanup;
        }
    }

    if ((error_code = lin_alg_qr_decompose(quantizer->Π)) != MATH_OPS_SUCCESS) 
            goto cleanup;

    /* lloyd max to get centroids */
    quantizer->book = codebook_lloyd_max(bit_width, dims, MAX_ITERATIONS);
    if (quantizer->book == NULL) { 
        error_code = QUANT_INIT_FAILED;
        goto cleanup;
    }

    quantizer->bit_width = bit_width;
    quantizer->dims = dims;

    quantizer->t_Π = lin_alg_transpose_matrix(quantizer->Π);
    if (quantizer->t_Π == NULL) {
        error_code = QUANT_INIT_FAILED;
        goto cleanup;
    }

    quantizer->t_S = lin_alg_transpose_matrix(quantizer->S);
    if (quantizer->t_S == NULL) {
        error_code = QUANT_INIT_FAILED;
        goto cleanup;
    }

cleanup:
    if (error_code != QUANT_SUCCESS) {
        turboquant_quantizer_destroy(&quantizer);
        return NULL;
    }

    return quantizer;
}

uint8_t turboquant_init(const size_t dims, const uint8_t bit_width) {
    if (dims == 0 || bit_width == 0)
        return QUANT_INIT_FAILED;

    mse_quantizer = turboquant_quantizer_init(dims, bit_width);
    if (mse_quantizer == NULL)
        return QUANT_INIT_FAILED;

    is_init = 1;
    return QUANT_SUCCESS;
}

void turboquant_clean() {
    if (mse_quantizer != NULL)
        turboquant_quantizer_destroy(&mse_quantizer);

    is_init = 0;
}


uint8_t turboquant_init_load(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) return QUANT_INIT_FAILED;

    // Clean any existing global quantizer before loading a new one
    if (mse_quantizer) turboquant_clean();

    size_t dims;
    uint8_t bit_width;

    // 1. Read Metadata
    if (fread(&dims, sizeof(size_t), 1, f) != 1) { fclose(f); return QUANT_INIT_FAILED; }
    if (fread(&bit_width, sizeof(uint8_t), 1, f) != 1) { fclose(f); return QUANT_INIT_FAILED; }

    // Allocate the container
    mse_quantizer = (turbo_quantizer*) malloc(sizeof(turbo_quantizer));
    if (!mse_quantizer) { fclose(f); return QUANT_INIT_FAILED; }

    mse_quantizer->dims = dims;
    mse_quantizer->bit_width = bit_width;

    // 2. Read Codebook
    mse_quantizer->book = (codebook*) malloc(sizeof(codebook));
    if (!mse_quantizer->book) { fclose(f); return QUANT_INIT_FAILED; }

    fread(&mse_quantizer->book->n_centroids, sizeof(size_t), 1, f);
    uint16_t n = mse_quantizer->book->n_centroids;
    mse_quantizer->book->centroids = (centroid*) malloc(sizeof(centroid) * n);
    fread(mse_quantizer->book->centroids, sizeof(centroid), n, f);

    // 3. Re-allocate and Read Matrices
    mse_quantizer->Π = lin_alg_create_matrix(dims, dims);
    mse_quantizer->t_Π = lin_alg_create_matrix(dims, dims);
    mse_quantizer->S = lin_alg_create_matrix(dims, dims);
    mse_quantizer->t_S = lin_alg_create_matrix(dims, dims); 

    for (size_t i = 0; i < dims; i++) {
        fread(&mse_quantizer->Π->matrix[i * mse_quantizer->Π->stride], sizeof(float), dims, f);
        fread(&mse_quantizer->t_Π->matrix[i * mse_quantizer->t_Π->stride], sizeof(float), dims, f);
        fread(&mse_quantizer->S->matrix[i * mse_quantizer->S->stride], sizeof(float), dims, f);
        fread(&mse_quantizer->t_S->matrix[i * mse_quantizer->t_S->stride], sizeof(float), dims, f);
    }

    fclose(f);
    is_init = 1;
    return QUANT_SUCCESS;
}

uint8_t turboquant_save(const char *filename) {
    if (!is_init || mse_quantizer == NULL) {
        return QUANT_UNINITIALIZED;
    }

    FILE *f = fopen(filename, "wb");
    if (!f) return QUANT_PROD_FAILED;

    // 1. Metadata: Dimensions and Bit-width
    fwrite(&mse_quantizer->dims, sizeof(size_t), 1, f);
    fwrite(&mse_quantizer->bit_width, sizeof(uint8_t), 1, f);

    // 2. Codebook: Centroid values and boundaries
    fwrite(&mse_quantizer->book->n_centroids, sizeof(size_t), 1, f);
    fwrite(mse_quantizer->book->centroids, sizeof(centroid), mse_quantizer->book->n_centroids, f);

    // 3. Matrices: Save all 4 pre-computed matrices
    for (size_t i = 0; i < mse_quantizer->dims; i++) {
        fwrite(&mse_quantizer->Π->matrix[i * mse_quantizer->Π->stride], sizeof(float), mse_quantizer->dims, f);
        fwrite(&mse_quantizer->t_Π->matrix[i * mse_quantizer->t_Π->stride], sizeof(float), mse_quantizer->dims, f);
        fwrite(&mse_quantizer->S->matrix[i * mse_quantizer->S->stride], sizeof(float), mse_quantizer->dims, f);
        fwrite(&mse_quantizer->t_S->matrix[i * mse_quantizer->t_S->stride], sizeof(float), mse_quantizer->dims, f);
    }

    fclose(f);
    return QUANT_SUCCESS;
}

quantization_result* turboquant_quantization_result_init() {
    quantization_result *results = (quantization_result*) malloc(sizeof(quantization_result));
    if (results == NULL)
        return NULL;

    results->bstring = NULL;
    results->qjl = NULL;
    results->residual_l2 = 0.0f;

    return results;
}

void turboquant_quantization_result_destroy(quantization_result **results) {
    if (results) {
        if ((*results)) {
            if ((*results)->bstring)
                free((*results)->bstring);
            if ((*results)->qjl)
                free((*results)->qjl);
            free((*results));
            *results = NULL;
        }
    }
}

float turboquant_mean_squared_error(const vector_t *vec_1, const vector_t *vec_2) {
    if (vec_1 == NULL || vec_2 == NULL) 
        return -1.0f;
    if (vec_1->vector == NULL || vec_2->vector == NULL || vec_1->n != vec_2->n)
        return -1.0f;

    size_t n = vec_1->n;
    size_t i = 0;
    float total_sum = 0.0f;

    if (n >= 8) {
        __m256 accum = _mm256_setzero_ps();

        for (; i + 7 < n; i += 8) {
            // Load 8 floats from each vector
            __m256 v1 = _mm256_load_ps(&vec_1->vector[i]);
            __m256 v2 = _mm256_load_ps(&vec_2->vector[i]);

            // Calculate difference: (v1 - v2)
            __m256 diff = _mm256_sub_ps(v1, v2);

            // Square and accumulate: accum += (diff * diff)
            // Using FMA (Fused Multiply-Add) is even faster if supported
            accum = _mm256_fmadd_ps(diff, diff, accum);
        }

        // Horizontal sum of the 8 lanes in the accumulator
        __m128 low128 = _mm256_castps256_ps128(accum);
        __m128 high128 = _mm256_extractf128_ps(accum, 1);
        __m128 sum128 = _mm_add_ps(low128, high128);

        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        total_sum = _mm_cvtss_f32(sum128);
    }

    // Scalar tail for remaining elements
    for (; i < n; i++) {
        float diff = vec_1->vector[i] - vec_2->vector[i];
        total_sum += (diff * diff);
    }

    return total_sum / (float)n;
}

/**
 * @param buffer: The destination byte array
 * @param dim_index: Which dimension we are saving (0, 1, 2...)
 * @param b: The bit width (e.g., 2)
 * @param value: The index of the centroid (e.g., 3)
 */
void pack_dynamic(uint8_t *buffer, size_t dim_index, uint8_t b, uint8_t value) {
    size_t total_bits = dim_index * b;
    size_t byte_pos = total_bits / 8;
    uint8_t bit_offset = total_bits % 8;

    // Mask the value to ensure it doesn't exceed b bits
    uint8_t clean_value = value & ((1 << b) - 1);

    // Shift the value to the correct starting position
    // We use a uint16_t temporary to handle cases where 
    // a value spans across two bytes
    uint16_t shifted_value = (uint16_t)clean_value << bit_offset;

    // Write to the current byte
    buffer[byte_pos] |= (uint8_t)(shifted_value & 0xFF);

    // If the value crosses into the next byte, write the remainder
    if (bit_offset + b > 8) {
        buffer[byte_pos + 1] |= (uint8_t)(shifted_value >> 8);
    }
}

uint8_t unpack_dynamic(const uint8_t *buffer, size_t dim_index, uint8_t b) {
    size_t total_bits = dim_index * b;
    size_t byte_pos = total_bits / 8;
    uint8_t bit_offset = total_bits % 8;

    // Read 16 bits to handle values that span across two bytes
    // We cast to uint16_t to safely shift bits from the next byte if needed
    uint16_t window = buffer[byte_pos];
    if (bit_offset + b > 8) {
        window |= (uint16_t)buffer[byte_pos + 1] << 8;
    }

    // Move the desired bits to the beginning and mask them
    return (uint8_t)((window >> bit_offset) & ((1 << b) - 1));
}

/* bstring buffer must be allocated with b-width * d prior to calling the function */
/* responsibility of the developer to zero out bstring */
uint8_t turboquant_mse_quantization(const vector_t *x, uint8_t *bstring) {
    if (x == NULL || bstring == NULL) 
        return QUANT_NULL;
    if (!is_init)
        return QUANT_UNINITIALIZED;

    /* Alloc rotated vector*/
    vector_t *y = lin_alg_create_vector(mse_quantizer->dims);
    if (y == NULL)
        return QUANT_NULL;

    vector_t *x_cpy = lin_alg_create_vector(mse_quantizer->dims);
    if (x_cpy == NULL) {
        lin_alg_free_vector(&y);
        return QUANT_NULL;
    }

    if (lin_alg_copy_vector(x_cpy, x) != SUCCESS) {
        lin_alg_free_vector(&x_cpy);
        lin_alg_free_vector(&y);
        return QUANT_MSE_FAILED;
    }

    /* L2 normalization on input vector */
    if (lin_alg_l2_normalize(x_cpy) != MATH_OPS_SUCCESS) {
        lin_alg_free_vector(&x_cpy);
        lin_alg_free_vector(&y);
        return QUANT_MSE_FAILED;
    }

    /* Apply rotation matrix on resulting vector */
    if (lin_alg_dot_productmv(mse_quantizer->Π, x_cpy, y) != SUCCESS) {
        lin_alg_free_vector(&x_cpy);
        lin_alg_free_vector(&y);
        return QUANT_MSE_FAILED;
    }

    /* Compute the mean squared difference */
    float current, min;
    size_t idx;
    for (int i = 0; i < mse_quantizer->dims; i++) {
        min = fabsf(y->vector[i] - mse_quantizer->book->centroids[0].value);
        idx = 0;

        for (int k = 1; k < mse_quantizer->book->n_centroids; k++) {
            current = fabsf(y->vector[i] - mse_quantizer->book->centroids[k].value);
            if (current < min) {
                min = current;
                idx = k;
            }

        }
        
        /* Now use that idx to pack it into bstring */
        pack_dynamic(bstring, i, mse_quantizer->bit_width, idx);
    }

    return QUANT_SUCCESS;
}

vector_t* turboquant_mse_dequantization(const uint8_t *bstring) {
    if (bstring == NULL)
        return NULL;
    if (!is_init)
        return NULL;

    uint8_t index;
    vector_t *y_hat = lin_alg_create_vector(mse_quantizer->dims);
    if (y_hat == NULL)
        return NULL;
    
    for (int i = 0; i < mse_quantizer->dims; i++) { 
        index = unpack_dynamic(bstring, i, mse_quantizer->bit_width); 
        y_hat->vector[i] = mse_quantizer->book->centroids[index].value; 
    }

    vector_t *x_hat = lin_alg_create_vector(mse_quantizer->dims);
    if (x_hat == NULL) {
        lin_alg_free_vector(&y_hat);
        return NULL;
    }

    if (lin_alg_dot_productmv(mse_quantizer->t_Π, y_hat, x_hat) != SUCCESS) {
        lin_alg_free_vector(&y_hat);
        lin_alg_free_vector(&x_hat);
        return NULL;
    }

    lin_alg_free_vector(&y_hat);
    return x_hat;
}

/* init_turboquant must be called with bit_width - 1 passed as a parameter */
uint8_t turboquant_prod_quantization(vector_t *x, quantization_result *results) {
    if (x == NULL)
        return QUANT_NULL;
    if (!is_init)
        return QUANT_UNINITIALIZED;

    size_t total_bytes = ((mse_quantizer->bit_width * mse_quantizer->dims) + 7) / 8; 
    uint8_t *bstring = (uint8_t*) malloc(total_bytes);
    if (bstring == NULL)
        return QUANT_PROD_FAILED;

    /* Zero out bstring */
    memset(bstring, 0, total_bytes);

    /* Apply MSE quantization */
    if (turboquant_mse_quantization(x, bstring) != QUANT_SUCCESS) {
        free(bstring);
        return QUANT_PROD_FAILED;
    }

    vector_t* x_hat = turboquant_mse_dequantization(bstring);
    if (x_hat == NULL) {
        free(bstring);
        return QUANT_PROD_FAILED;
    }

    vector_t *residual = lin_alg_create_vector(mse_quantizer->dims);
    if (residual == NULL) {
        free(bstring);
        lin_alg_free_vector(&x_hat);
        return QUANT_PROD_FAILED;
    }

    if (lin_alg_copy_vector(residual, x) != SUCCESS) {
        free(bstring);
        lin_alg_free_vector(&x_hat);
        lin_alg_free_vector(&residual);
        return QUANT_PROD_FAILED;
    }

    if (lin_alg_l2_normalize(residual) != MATH_OPS_SUCCESS) {
        free(bstring);
        lin_alg_free_vector(&x_hat);
        lin_alg_free_vector(&residual);
        return QUANT_PROD_FAILED; 
    }

    if (lin_alg_sub_vectors(residual, x_hat) != SUCCESS) {
        free(bstring);
        lin_alg_free_vector(&x_hat);
        lin_alg_free_vector(&residual);
        return QUANT_PROD_FAILED;
    }

    vector_t *result = lin_alg_create_vector(mse_quantizer->dims);
    if (result == NULL) {
        free(bstring);
        lin_alg_free_vector(&x_hat);
        lin_alg_free_vector(&residual);
        return QUANT_PROD_FAILED;
    }

    if (lin_alg_dot_productmv(mse_quantizer->S, residual, result) != SUCCESS) {
        free(bstring);
        lin_alg_free_vector(&x_hat);
        lin_alg_free_vector(&residual);
        lin_alg_free_vector(&result);
        return QUANT_PROD_FAILED;
    }

    size_t qjl_bytes = mse_quantizer->dims / 8;
    uint8_t *qjl_packed = (uint8_t *) malloc(qjl_bytes);
    if (qjl_packed == NULL) {
        free(bstring);
        lin_alg_free_vector(&x_hat);
        lin_alg_free_vector(&residual);
        lin_alg_free_vector(&result);
        return QUANT_PROD_FAILED;
    }
    memset(qjl_packed, 0, qjl_bytes);

    /* Put in qjl vector the signs of the resulting vector */
    for (int i = 0; i < mse_quantizer->dims; i++) {
        uint8_t sign_bit = (result->vector[i] < 0.0f) ? 1 : 0;
        pack_dynamic(qjl_packed, i, 1, sign_bit);
    }

    results->qjl = qjl_packed;
    results->bstring = bstring;
    results->residual_l2 = lin_alg_l2(residual);

    lin_alg_free_vector(&x_hat);
    lin_alg_free_vector(&result);
    lin_alg_free_vector(&residual);
    return QUANT_SUCCESS;
}

/* Implements Step 9-12 of Algorithm 2: DEQUANT_prod */
vector_t* turboquant_prod_dequantization(const quantization_result *res) {
    if (res == NULL || !is_init) return NULL;

    const size_t d = mse_quantizer->dims;

    /* Step 10: Reconstruct the primary MSE component (x_mse_tilde) */
    /* */
    vector_t *x_mse = turboquant_mse_dequantization(res->bstring);
    if (x_mse == NULL) return NULL;

    /* Step 11: Reconstruct the residual component (x_qj1_tilde) */
    /* */
    vector_t *qj1_floats = lin_alg_create_vector(d);
    if (qj1_floats == NULL) {
        lin_alg_free_vector(&x_mse);
        return NULL;
    }
    
    for (size_t i = 0; i < d; i++) {
        uint8_t sign_bit = unpack_dynamic(res->qjl, i, 1);

        qj1_floats->vector[i] = (sign_bit == 1) ? -1.0f : 1.0f;
    }


    vector_t *x_qj1 = lin_alg_create_vector(d);
    if (x_qj1 == NULL) {
        lin_alg_free_vector(&x_mse);
        lin_alg_free_vector(&qj1_floats);
        return NULL;
    }

    /* Apply the transpose of the Gaussian matrix S */
    /* */
    if (lin_alg_dot_productmv(mse_quantizer->t_S, qj1_floats, x_qj1) != SUCCESS) {
        lin_alg_free_vector(&x_mse);
        lin_alg_free_vector(&qj1_floats);
        lin_alg_free_vector(&x_qj1);
        return NULL;
    }

    /* Apply the scaling factor: (sqrt(pi/2) / d) * gamma */
    /* */
    float scale = (sqrtf(PI / 2.0f) / (float)d) * res->residual_l2;
    if (lin_alg_scale_vector(x_qj1, scale) != SUCCESS) {
        lin_alg_free_vector(&x_mse);
        lin_alg_free_vector(&qj1_floats);
        lin_alg_free_vector(&x_qj1);
        return NULL;
    }

    /* Step 12: Sum both components to get final reconstruction */
    /* */
    if (lin_alg_add_vectors(x_mse, x_qj1) != SUCCESS) {
        lin_alg_free_vector(&x_mse);
        lin_alg_free_vector(&qj1_floats);
        lin_alg_free_vector(&x_qj1);
        return NULL;
    }

    /* Clean up temporary buffers */
    lin_alg_free_vector(&qj1_floats);
    lin_alg_free_vector(&x_qj1);

    return x_mse; 
}
