#include "../include/turboquant.h"
#include "../include/math_ops.h"
#include "../include/errors.h"
#include "../include/config.h"
#include "../include/lin_alg.h"
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

turbo_quantizer *mse_quantizer = NULL;
uint8_t is_init = 0;

void destroy_quantizer(turbo_quantizer **quantizer) {
    if (quantizer) {
        if ((*quantizer)) {
            if ((*quantizer)->book) 
                destroy_codebook(&(*quantizer)->book);
            if((*quantizer)->Π)
                free_mtx((*quantizer)->Π, (*quantizer)->dims);
            if ((*quantizer)->t_Π)
                free_mtx((*quantizer)->t_Π, (*quantizer)->dims);
            if ((*quantizer)->S)
                free_mtx((*quantizer)->S, (*quantizer)->dims);
            if ((*quantizer)->t_S)
                free_mtx((*quantizer)->t_S, (*quantizer)->dims);

            free((*quantizer));
            *quantizer = NULL;
        }
    }
}

turbo_quantizer* init_quantizer(const size_t dims, const uint8_t bit_width) {
   
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

    quantizer->S = normal_distribution_random_matrix(dims);
    if (quantizer->S == NULL) {
        error_code = QUANT_INIT_FAILED;
        goto cleanup;
    }

    quantizer->Π = create_mtx(dims, dims);
    if (quantizer->Π == NULL) {
        error_code = QUANT_INIT_FAILED;
        goto cleanup;
    }

    for (int i = 0; i < dims; i++) {
        if (copy_vec(quantizer->Π[i], quantizer->S[i], dims) != SUCCESS) {
            error_code = QUANT_INIT_FAILED;
            goto cleanup;
        }
    }

    if ((error_code = qr_decomposition(quantizer->Π , dims)) != MATH_OPS_SUCCESS) 
            goto cleanup;

    /* lloyd max to get centroids */
    quantizer->book = lloyd_max(bit_width, dims, MAX_ITERATIONS);
    if (quantizer->book == NULL) { 
        error_code = QUANT_INIT_FAILED;
        goto cleanup;
    }

    quantizer->bit_width = bit_width;
    quantizer->dims = dims;

    quantizer->t_Π = transpose_mtx((const float**) quantizer->Π, dims, dims);
    if (quantizer->t_Π == NULL) {
        error_code = QUANT_INIT_FAILED;
        goto cleanup;
    }

    quantizer->t_S = transpose_mtx((const float**) quantizer->S, dims, dims);
    if (quantizer->t_S == NULL) {
        error_code = QUANT_INIT_FAILED;
        goto cleanup;
    }

cleanup:
    if (error_code != QUANT_SUCCESS) {
        destroy_quantizer(&quantizer);
        return NULL;
    }

    return quantizer;
}

uint8_t init_turboquant(const size_t dims, const uint8_t bit_width) {
    if (dims == 0 || bit_width == 0)
        return QUANT_INIT_FAILED;

    mse_quantizer = init_quantizer(dims, bit_width);
    if (mse_quantizer == NULL)
        return QUANT_INIT_FAILED;

    is_init = 1;
    return QUANT_SUCCESS;
}

void clean_turboquant() {
    if (mse_quantizer != NULL)
        destroy_quantizer(&mse_quantizer);

    is_init = 0;
}


uint8_t init_load_turboquant(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) return QUANT_INIT_FAILED;

    // Clean any existing global quantizer before loading a new one
    if (mse_quantizer) clean_turboquant();

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
    mse_quantizer->Π = create_mtx(dims, dims);
    mse_quantizer->t_Π = create_mtx(dims, dims);
    mse_quantizer->S = create_mtx(dims, dims);
    mse_quantizer->t_S = create_mtx(dims, dims);

    for (size_t i = 0; i < dims; i++) {
        fread(mse_quantizer->Π[i], sizeof(float), dims, f);
        fread(mse_quantizer->t_Π[i], sizeof(float), dims, f);
        fread(mse_quantizer->S[i], sizeof(float), dims, f);
        fread(mse_quantizer->t_S[i], sizeof(float), dims, f);
    }

    fclose(f);
    is_init = 1;
    return QUANT_SUCCESS;
}

uint8_t save_turboquant(const char *filename) {
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
        fwrite(mse_quantizer->Π[i], sizeof(float), mse_quantizer->dims, f);
        fwrite(mse_quantizer->t_Π[i], sizeof(float), mse_quantizer->dims, f);
        fwrite(mse_quantizer->S[i], sizeof(float), mse_quantizer->dims, f);
        fwrite(mse_quantizer->t_S[i], sizeof(float), mse_quantizer->dims, f);
    }

    fclose(f);
    return QUANT_SUCCESS;
}

quantization_result* init_quantization_result() {
    quantization_result *results = (quantization_result*) malloc(sizeof(quantization_result));
    if (results == NULL)
        return NULL;

    results->bstring = NULL;
    results->qjl = NULL;
    results->residual_l2 = 0.0f;

    return results;
}

void destroy_quantization_result(quantization_result **results) {
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


float mean_squared_error(const float *vec_1, const float *vec_2, const size_t d) {
    if (vec_1 == NULL || vec_2 == NULL || d == 0) {
        return -1.0f;
    }

    float sum_squared_diff = 0.0f; 

    for (size_t i = 0; i < d; ++i) {
        float diff = vec_1[i] - vec_2[i];
        sum_squared_diff += (double)(diff * diff);
    }

    return (float)(sum_squared_diff / (float)d);
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
uint8_t mse_quantization(const float *x, uint8_t *bstring) {
    if (x == NULL || bstring == NULL) 
        return QUANT_NULL;
    if (!is_init)
        return QUANT_UNINITIALIZED;

    /* Alloc rotated vector*/
    float *y = create_vec(mse_quantizer->dims);
    if (y == NULL)
        return QUANT_NULL;

    float *x_cpy = create_vec(mse_quantizer->dims);
    if (x_cpy == NULL) {
        free_vec(y);
        return QUANT_NULL;
    }

    if (copy_vec(x_cpy, x, mse_quantizer->dims) != SUCCESS) {
        free_vec(x_cpy);
        free_vec(y);
        return QUANT_MSE_FAILED;
    }

    /* L2 normalization on input vector */
    if (l2_normalization(x_cpy, mse_quantizer->dims) != MATH_OPS_SUCCESS) {
        free_vec(x_cpy);
        free_vec(y);
        return QUANT_MSE_FAILED;
    }

    /* Apply rotation matrix on resulting vector */
    if (dot_product_mtx_vec((const float**)mse_quantizer->Π, x_cpy, mse_quantizer->dims,
                mse_quantizer->dims, mse_quantizer->dims, y) != SUCCESS) {
        free_vec(x_cpy);
        free_vec(y);
        return QUANT_MSE_FAILED;
    }

    /* Compute the mean squared difference */
    float current, min;
    size_t idx;
    for (int i = 0; i < mse_quantizer->dims; i++) {
        min = fabsf(y[i] - mse_quantizer->book->centroids[0].value);
        idx = 0;

        for (int k = 1; k < mse_quantizer->book->n_centroids; k++) {
            current = fabsf(y[i] - mse_quantizer->book->centroids[k].value);
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

float* mse_dequantization(const uint8_t *bstring) {
    if (bstring == NULL)
        return NULL;
    if (!is_init)
        return NULL;

    uint8_t index;
    float *y_hat = create_vec(mse_quantizer->dims);
    if (y_hat == NULL)
        return NULL;
    
    for (int i = 0; i< mse_quantizer->dims; i++) { 
        index = unpack_dynamic(bstring, i, mse_quantizer->bit_width); 
        y_hat[i] = mse_quantizer->book->centroids[index].value; 
    }

    float *x_hat = create_vec(mse_quantizer->dims);
    if (x_hat == NULL) {
        free_vec(y_hat);
        return NULL;
    }

    if (dot_product_mtx_vec((const float**) mse_quantizer->t_Π, y_hat, mse_quantizer->dims,
                mse_quantizer->dims, mse_quantizer->dims, x_hat) != SUCCESS) {
        free_vec(y_hat);
        free_vec(x_hat);
        return NULL;
    }

    free_vec(y_hat);
    return x_hat;
}

/* init_turboquant must be called with bit_width - 1 passed as a parameter */
uint8_t prod_quantization(float *x, quantization_result *results) {
    if (x == NULL)
        return QUANT_NULL;
    if (!is_init)
        return QUANT_UNINITIALIZED;

    size_t total_bytes = (mse_quantizer->bit_width * mse_quantizer->dims) / 8; 
    uint8_t *bstring = (uint8_t*) malloc(total_bytes);
    if (bstring == NULL)
        return QUANT_PROD_FAILED;

    /* Zero out bstring */
    memset(bstring, 0, total_bytes);

    /* Apply MSE quantization */
    if (mse_quantization(x, bstring) != QUANT_SUCCESS) {
        free(bstring);
        return QUANT_PROD_FAILED;
    }

    float* x_hat = mse_dequantization(bstring);
    if (x_hat == NULL) {
        free(bstring);
        return QUANT_PROD_FAILED;
    }

    float *residual = create_vec(mse_quantizer->dims);
    if (residual == NULL) {
        free(bstring);
        free_vec(x_hat);
        return QUANT_PROD_FAILED;
    }

    if (copy_vec(residual, x, mse_quantizer->dims) != SUCCESS) {
        free(bstring);
        free_vec(x_hat);
        free_vec(residual);
        return QUANT_PROD_FAILED;
    }

    if (l2_normalization(residual, mse_quantizer->dims) != MATH_OPS_SUCCESS) {
        free(bstring);
        free_vec(x_hat);
        free_vec(residual);
        return QUANT_PROD_FAILED; 
    }

    if (sub_vec(residual, x_hat, mse_quantizer->dims) != SUCCESS) {
        free(bstring);
        free_vec(x_hat);
        free_vec(residual);
        return QUANT_PROD_FAILED;
    }

    float *result = create_vec(mse_quantizer->dims);
    if (result == NULL) {
        free(bstring);
        free_vec(x_hat);
        free_vec(residual);
        return QUANT_PROD_FAILED;
    }

    if (dot_product_mtx_vec((const float**) mse_quantizer->S, residual, mse_quantizer->dims, 
                mse_quantizer->dims, mse_quantizer->dims, result) != SUCCESS) {
        free(bstring);
        free_vec(x_hat);
        free_vec(residual);
        free_vec(result);
        return QUANT_PROD_FAILED;
    }

    size_t qjl_bytes = mse_quantizer->dims / 8;
    uint8_t *qjl_packed = (uint8_t *) malloc(qjl_bytes);
    if (qjl_packed == NULL) {
        free(bstring);
        free_vec(x_hat);
        free_vec(residual);
        free_vec(result);
        return QUANT_PROD_FAILED;
    }
    memset(qjl_packed, 0, qjl_bytes);

    /* Put in qjl vector the signs of the resulting vector */
    for (int i = 0; i < mse_quantizer->dims; i++) {
        uint8_t sign_bit = (result[i] < 0.0f) ? 1 : 0;
        pack_dynamic(qjl_packed, i, 1, sign_bit);
    }

    results->qjl = qjl_packed;
    results->bstring = bstring;
    results->residual_l2 = compute_l2_norm(residual, mse_quantizer->dims);

    free_vec(x_hat);
    free_vec(result);
    free_vec(residual);
    return QUANT_SUCCESS;
}

/* Implements Step 9-12 of Algorithm 2: DEQUANT_prod */
float* prod_dequantization(const quantization_result *res) {
    if (res == NULL || !is_init) return NULL;

    const size_t d = mse_quantizer->dims;

    /* Step 10: Reconstruct the primary MSE component (x_mse_tilde) */
    /* */
    float *x_mse = mse_dequantization(res->bstring);
    if (x_mse == NULL) return NULL;

    /* Step 11: Reconstruct the residual component (x_qj1_tilde) */
    /* */
    float *qj1_floats = create_vec(d);
    if (qj1_floats == NULL) {
        free_vec(x_mse);
        return NULL;
    }
    
    for (size_t i = 0; i < d; i++) {
        uint8_t sign_bit = unpack_dynamic(res->qjl, i, 1);

        qj1_floats[i] = (sign_bit == 1) ? -1.0f : 1.0f;
    }


    /* Convert the binary qjl values back to floats {-1, 1} */
    /* Note: If you pack these into bits later, you'll use unpack_dynamic here */
    for (size_t i = 0; i < d; i++) {
        // Mapping: 1 (negative) -> -1.0f, 0 (positive) -> 1.0f
        qj1_floats[i] = (res->qjl[i] == 1) ? -1.0f : 1.0f;
    }

    float *x_qj1 = create_vec(d);
    if (x_qj1 == NULL) {
        free_vec(x_mse);
        free_vec(qj1_floats);
        return NULL;
    }

    /* Apply the transpose of the Gaussian matrix S */
    /* */
    if (dot_product_mtx_vec((const float**)mse_quantizer->t_S, qj1_floats, d, d, d, x_qj1) != SUCCESS) {
        free_vec(x_mse);
        free_vec(qj1_floats);
        free_vec(x_qj1);
        return NULL;
    }

    /* Apply the scaling factor: (sqrt(pi/2) / d) * gamma */
    /* */
    float scale = (sqrtf(PI / 2.0f) / (float)d) * res->residual_l2;
    if (scale_vec(x_qj1, scale, d) != SUCCESS) {
        free_vec(x_mse);
        free_vec(qj1_floats);
        free_vec(x_qj1);
        return NULL;
    }

    /* Step 12: Sum both components to get final reconstruction */
    /* */
    if (add_vec(x_mse, x_qj1, d) != SUCCESS) {
        free_vec(x_mse);
        free_vec(qj1_floats);
        free_vec(x_qj1);
        return NULL;
    }

    /* Clean up temporary buffers */
    free_vec(qj1_floats);
    free_vec(x_qj1);

    return x_mse; 
}
