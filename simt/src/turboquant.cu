#include "../include/turboquant.h"
#include "../include/codebook.h"
#include "../include/errors.h"
#include "../include/config.h"
#include "../include/lin_alg.h"
#include <complex.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// --- FORENSICS MACRO ---
#define FORENSIC_LOG(step_name, cuda_err) do { \
    FILE *f_log = fopen("simt_forensics.log", "a"); \
    if (f_log) { \
        fprintf(f_log, "[FORENSICS] %s | Code: %d | Message: %s\n", \
                step_name, cuda_err, cudaGetErrorString(cuda_err)); \
        fclose(f_log); \
    } \
} while(0)
// -----------------------

void turboquant_quantizer_destroy(turbo_quantizer **quantizer) {
    if (quantizer) {
        if ((*quantizer)) {
            if ((*quantizer)->book) 
                codebook_destroy(&(*quantizer)->book);
            if((*quantizer)->Π)
                lin_alg_free_matrix(&(*quantizer)->Π);
            if ((*quantizer)->S)
                lin_alg_free_matrix(&(*quantizer)->S);
            if ((*quantizer)->d_centroids)
                cudaFree((*quantizer)->d_centroids);
        }

            free((*quantizer));
            *quantizer = NULL;
    }
}

static uint8_t sync_centroids(turbo_quantizer *q) {
    if (!q || !q->book) return QUANT_NULL;
    
    size_t n = q->book->n_centroids;

    if (q->d_centroids != NULL) {
        cudaFree(q->d_centroids);
        q->d_centroids = NULL;
    }
    
    if (cudaMalloc(&q->d_centroids, n * sizeof(float)) != cudaSuccess)
        return QUANT_NULL;
    
    float *h_temp = (float*)malloc(n * sizeof(float));
    if (h_temp == NULL) {
        cudaFree(q->d_centroids);
        return QUANT_NULL;
    }

    for(size_t i=0; i<n; i++) h_temp[i] = q->book->centroids[i].value;
    
    if (cudaMemcpy(q->d_centroids, h_temp, n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(q->d_centroids);
        free(h_temp);
        return QUANT_NULL;
    }

    free(h_temp);
    return QUANT_SUCCESS;
}

turbo_quantizer* turboquant_quantizer_init(const size_t dims, const uint8_t bit_width) {
   
    uint8_t error_code = QUANT_SUCCESS;
    
    /* Initialize the Haar Matrix */
    turbo_quantizer *quantizer = (turbo_quantizer*) malloc(sizeof(turbo_quantizer));
    if (quantizer == NULL) 
        return NULL;

    quantizer->book = NULL;
    quantizer->Π = NULL;
    quantizer->d_centroids = NULL;
    quantizer->bit_width = bit_width;
    quantizer->dims = dims;


    quantizer->S = lin_alg_create_matrix(dims, dims);
    if (quantizer->S == NULL){
        free(quantizer);
        return NULL;
    }

    quantizer->Π = lin_alg_create_matrix(dims, dims);
    if (quantizer->Π == NULL) {
        error_code = QUANT_INIT_FAILED;
        goto cleanup;
    }

    quantizer->S = lin_alg_normal_rand_matrix(quantizer->S);
    if (quantizer->S == NULL) {
        error_code = QUANT_INIT_FAILED;
        goto cleanup;
    }
    
    if (lin_alg_copy_matrix(quantizer->Π, quantizer->S) != SUCCESS) {
        error_code = QUANT_INIT_FAILED;
        goto cleanup;
    }

    if (lin_alg_qr_decompose(quantizer->Π) != MATH_OPS_SUCCESS) {
        error_code = QUANT_INIT_FAILED;
        goto cleanup;
    }

    /* lloyd max to get centroids */
    quantizer->book = codebook_lloyd_max(bit_width, dims, MAX_ITERATIONS);
    if (quantizer->book == NULL) { 
        error_code = QUANT_INIT_FAILED;
        goto cleanup;
    }

    if (sync_centroids(quantizer) != QUANT_SUCCESS) {
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

uint8_t turboquant_init(turboquant_context_t **context, const size_t dims,
        const uint8_t bit_width) {
    if (dims == 0 || bit_width == 0)
        return QUANT_INIT_FAILED;

    *context = (turboquant_context_t*) malloc(sizeof(turboquant_context_t));
    if (*context == NULL)
        return QUANT_INIT_FAILED;

    (*context)->mse_quantizer = NULL, (*context)->mse_buffer = NULL, (*context)->y = NULL,
        (*context)->h_bstring = NULL, (*context)->d_bstring = NULL, (*context)->h_qjl = NULL,
        (*context)->d_qjl = NULL, (*context)->compute_stream = NULL;

    (*context)->mse_quantizer = turboquant_quantizer_init(dims, bit_width);
    if ((*context)->mse_quantizer == NULL) {
        turboquant_clean(*context);
        free(*context);
        return QUANT_INIT_FAILED;
    }

    (*context)->mse_buffer = lin_alg_create_vector(dims);
    if ((*context)->mse_buffer == NULL) {
        turboquant_clean(*context);
        free(*context);
        return QUANT_INIT_FAILED;
    }
    
    (*context)->y = lin_alg_create_vector(dims);
    if ((*context)->y == NULL) {
        turboquant_clean(*context);
        free(*context);
        return QUANT_INIT_FAILED;
    }

    /* ----------- Byte Array Allocation on GPU and CPU ---------- */
    size_t b_size = ((dims * bit_width + 31) / 32) * 4;
    cudaHostAlloc((void**)&(*context)->h_bstring, b_size, cudaHostAllocMapped);
    
    cudaError_t err_b = cudaHostGetDevicePointer((void**)&(*context)->d_bstring, (void*)(*context)->h_bstring, 0);
    FORENSIC_LOG("turboquant_init -> bstring mapping", err_b);
    
    (*context)->bstring_size = b_size;
    memset((*context)->h_bstring, 0, b_size);
    /* ---------------------------------------------------------- */

    /* ----------- QJL Byte Array Allocation on GPU and CPU ---------- */
    size_t qjl_size = ((dims + 31) / 32) * 4;
    cudaHostAlloc((void**)&(*context)->h_qjl, qjl_size, cudaHostAllocMapped);
    
    cudaError_t err_q = cudaHostGetDevicePointer((void**)&(*context)->d_qjl, (void*)(*context)->h_qjl, 0);
    FORENSIC_LOG("turboquant_init -> qjl mapping", err_q);

    (*context)->qjl_size = qjl_size;
    memset((*context)->h_qjl, 0, qjl_size);
    /* ---------------------------------------------------------- */
    
    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        turboquant_clean(*context);
        free(*context);
        return QUANT_INIT_FAILED;
    }

    (*context)->compute_stream = (void *)stream;
    if (lin_alg_set_stream((*context)->compute_stream) != SUCCESS) {
        turboquant_clean(*context);
        free(*context);
        return QUANT_INIT_FAILED;
    }

    (*context)->is_init = 1;
    return QUANT_SUCCESS;
}

void turboquant_context_destroy(turboquant_context_t **context) {
    if (context) {
        turboquant_clean(*context);
        lin_alg_runtime_shutdown();
        free(*context);
        *context = NULL;
    }
}

void turboquant_clean(turboquant_context_t *context) {
    if (context == NULL)
        return;

    if (context->mse_quantizer != NULL)
        turboquant_quantizer_destroy(&context->mse_quantizer);

    if (context->mse_buffer != NULL)
        lin_alg_free_vector(&context->mse_buffer);
 
    if (context->y != NULL)
        lin_alg_free_vector(&context->y);

    // 1. Clean the Stream
    if (context->compute_stream != NULL) {
        cudaStream_t stream = (cudaStream_t)context->compute_stream;
        cudaStreamSynchronize(stream); // Finish pending Llama heads
        cudaStreamDestroy(stream);
        context->compute_stream = NULL;
    }

    // 2. Clean the Zero-Copy bstring
    if (context->h_bstring != NULL) {
        // This one call cleans up both h_bstring AND the d_bstring mapping
        cudaFreeHost(context->h_bstring);
        context->h_bstring = NULL;
        context->d_bstring = NULL;
    }

    if (context->h_qjl != NULL) {
        cudaFreeHost(context->h_qjl);
        context->h_qjl = NULL;
        context->d_qjl = NULL;
    }

    context->is_init = 0;
}

uint8_t turboquant_init_load(turboquant_context_t *context, const char *filename) {
    if (filename == NULL)
        return QUANT_INIT_FAILED;

    uint8_t error_code = QUANT_SUCCESS;

    FILE *f = fopen(filename, "rb");
    if (!f) return QUANT_INIT_FAILED;

    // 1. Cleanup existing global quantizer
    if (context->mse_quantizer) turboquant_quantizer_destroy(&context->mse_quantizer);

    // 2. Metadata: dims, bit_width, and the SAVED stride
    size_t dims, saved_stride, qjl_size, b_size;
    uint8_t bit_width;
    uint16_t n;
    float *row_buffer = NULL;

    if (fread(&dims, sizeof(size_t), 1, f) != 1) { error_code = QUANT_INIT_FAILED; goto cleanup; }
    if (fread(&bit_width, sizeof(uint8_t), 1, f) != 1) { error_code = QUANT_INIT_FAILED; goto cleanup; }
    if (fread(&saved_stride, sizeof(size_t), 1, f) != 1) { error_code = QUANT_INIT_FAILED; goto cleanup; }

    if (context->mse_buffer && context->mse_buffer->n != dims) {
        lin_alg_free_vector(&context->mse_buffer);
    }
    // Allocate container
    if (!context->mse_buffer) {
        context->mse_buffer = lin_alg_create_vector(dims);
        if (context->mse_buffer == NULL) { error_code = QUANT_INIT_FAILED; goto cleanup; }
    } if (!context->y) {
        context->y = lin_alg_create_vector(dims);
        if (context->y == NULL) { error_code = QUANT_INIT_FAILED; goto cleanup; }
    } 

    context->mse_quantizer = (turbo_quantizer*) malloc(sizeof(turbo_quantizer));
    if (!context->mse_quantizer) { error_code = QUANT_INIT_FAILED; goto cleanup; }
    
    context->mse_quantizer->dims = dims;
    context->mse_quantizer->bit_width = bit_width;
   
    if (!context->compute_stream) {
        cudaStream_t stream;
        if (cudaStreamCreate(&stream) != cudaSuccess) { error_code = QUANT_INIT_FAILED; goto cleanup; };

        context->compute_stream = (void *)stream;
    }
    if (lin_alg_set_stream(context->compute_stream) != SUCCESS) { error_code = QUANT_INIT_FAILED; goto cleanup; }
    
    b_size = ((dims * bit_width + 31) / 32) * 4;
    if (context->h_bstring) {
        if (context->bstring_size != b_size) {
            cudaFreeHost(context->h_bstring);
            cudaHostAlloc((void**)&context->h_bstring, b_size, cudaHostAllocMapped);
            
            cudaError_t err_load_b1 = cudaHostGetDevicePointer((void**)&context->d_bstring, (void*)context->h_bstring, 0);
            FORENSIC_LOG("turboquant_init_load (realloc) -> bstring mapping", err_load_b1);

            context->bstring_size = b_size;
        }
    } else {
            cudaHostAlloc((void**)&context->h_bstring, b_size, cudaHostAllocMapped);

            cudaError_t err_load_b2 = cudaHostGetDevicePointer((void**)&context->d_bstring, (void*)context->h_bstring, 0);
            FORENSIC_LOG("turboquant_init_load (fresh) -> bstring mapping", err_load_b2);

            context->bstring_size = b_size;
    }

    memset(context->h_bstring, 0, b_size);

    qjl_size = ((dims + 31) / 32) * 4;
    if (context->h_qjl) {
        if (context->qjl_size != qjl_size) {
            cudaFreeHost(context->h_qjl);
            cudaHostAlloc((void**)&context->h_qjl, qjl_size, cudaHostAllocMapped);

            cudaError_t err_load_q1 = cudaHostGetDevicePointer((void**)&context->d_qjl, (void*)context->h_qjl, 0);
            FORENSIC_LOG("turboquant_init_load (realloc) -> qjl mapping", err_load_q1);
            
            context->qjl_size = qjl_size;
        }
    } else {
        cudaHostAlloc((void**)&context->h_qjl, qjl_size, cudaHostAllocMapped);
        
        cudaError_t err_load_q2 = cudaHostGetDevicePointer((void**)&context->d_qjl, (void*)context->h_qjl, 0);
        FORENSIC_LOG("turboquant_init_load (fresh) -> qjl mapping", err_load_q2);

        context->qjl_size = qjl_size;
    }
    memset(context->h_qjl, 0, qjl_size);

    // 3. Load Codebook (POD centroids)
    context->mse_quantizer->book = (codebook*) malloc(sizeof(codebook));
    if (fread(&context->mse_quantizer->book->n_centroids, sizeof(size_t), 1, f) != 1) { error_code = QUANT_INIT_FAILED; goto cleanup; }
     
    n = (uint16_t)context->mse_quantizer->book->n_centroids;
    context->mse_quantizer->book->centroids = (centroid*) malloc(sizeof(centroid) * n);
    if (context->mse_quantizer->book->centroids == NULL) { error_code = QUANT_INIT_FAILED; goto cleanup; }
    if (fread(context->mse_quantizer->book->centroids, sizeof(centroid), n, f) != n) { error_code = QUANT_INIT_FAILED; goto cleanup; }
    if (sync_centroids(context->mse_quantizer) != QUANT_SUCCESS) { error_code = QUANT_INIT_FAILED; goto cleanup; }

    // 4. Allocate Matrices on GPU (This run's optimal stride)
    if ((context->mse_quantizer->Π = lin_alg_create_matrix(dims, dims)) == NULL) {error_code = QUANT_INIT_FAILED; goto cleanup; }
    if ((context->mse_quantizer->S = lin_alg_create_matrix(dims, dims)) == NULL) { error_code = QUANT_INIT_FAILED; goto cleanup; }

    // 5. Row-by-row Matrix Transfer (Handling Stride Mismatch)
    /* We use a buffer to read exactly one row as stored in the file */
    row_buffer = (float*) malloc(saved_stride * sizeof(float));
    if (row_buffer == NULL) { error_code = QUANT_INIT_FAILED; goto cleanup; }

    // Load Matrix Π
    for (size_t i = 0; i < dims; i++) {
        if (fread(row_buffer, sizeof(float), saved_stride, f) != saved_stride) { error_code = QUANT_INIT_FAILED; goto cleanup; }
        // Copy only the valid data (dims) to the GPU at the current stride offset
        if (cudaMemcpy((float*)context->mse_quantizer->Π->matrix + (i * context->mse_quantizer->Π->stride), 
                   row_buffer, dims * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) { error_code = QUANT_INIT_FAILED; goto cleanup; }
    }

    // Load Matrix S (using the same logic)
    for (size_t i = 0; i < dims; i++) {
        if (fread(row_buffer, sizeof(float), saved_stride, f) != saved_stride) { error_code = QUANT_INIT_FAILED; goto cleanup; }
        if (cudaMemcpy((float*)context->mse_quantizer->S->matrix + (i * context->mse_quantizer->S->stride), 
                   row_buffer, dims * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {error_code = QUANT_INIT_FAILED; goto cleanup; }
    }


cleanup:
    if (row_buffer) free(row_buffer);
    fclose(f);

    if (error_code == QUANT_SUCCESS)
        context->is_init = 1;
    if (error_code != QUANT_SUCCESS) {
        turboquant_quantizer_destroy(&context->mse_quantizer);
        lin_alg_free_vector(&context->mse_buffer);
        lin_alg_free_vector(&context->y);
    }

    return error_code;
}

uint8_t turboquant_save(turboquant_context_t *context, const char *filename) {
    if (filename == NULL || context == NULL)
        return QUANT_INIT_FAILED;
    else if (!context->is_init || context->mse_quantizer == NULL) {
        return QUANT_UNINITIALIZED;
    }

    FILE *f = fopen(filename, "wb");
    if (!f) return QUANT_PROD_FAILED;

    // 1. Metadata: Dimensions, Bit-width, AND the Stride
    // We save the stride so the 'load' function knows exactly how the file is padded.
    fwrite(&context->mse_quantizer->dims, sizeof(size_t), 1, f);
    fwrite(&context->mse_quantizer->bit_width, sizeof(uint8_t), 1, f);
    
    // Crucial: Save the stride of the matrices we are about to write
    size_t current_stride = context->mse_quantizer->Π->stride;
    fwrite(&current_stride, sizeof(size_t), 1, f);

    // 2. Codebook: Centroid values (POD struct)
    fwrite(&context->mse_quantizer->book->n_centroids, sizeof(size_t), 1, f);
    fwrite(context->mse_quantizer->book->centroids, sizeof(centroid), context->mse_quantizer->book->n_centroids, f);

    // 3. Matrix Serialization (Device to Host to File)
    // We allocate one temporary CPU buffer to pull the matrices from the GPU.
    size_t matrix_size_bytes = context->mse_quantizer->dims * current_stride * sizeof(float);
    float *h_buffer = (float*)malloc(matrix_size_bytes);
    
    if (!h_buffer) { fclose(f); return QUANT_PROD_FAILED; }

    // Helper array of the matrices we need to save
    // Note: Since you're using Llama 3.1, Π and S are the core ones.
    matrix_t* targets[] = {context->mse_quantizer->Π, context->mse_quantizer->S};
    int num_targets = 2;

    for (int i = 0; i < num_targets; i++) {
        matrix_t *mtx = targets[i];
        
        // Pull the entire strided matrix from GPU to CPU in one go
        cudaMemcpy2D(h_buffer, current_stride * sizeof(float), 
                     mtx->matrix, mtx->stride * sizeof(float), 
                     mtx->n * sizeof(float), mtx->m, 
                     cudaMemcpyDeviceToHost);

        // Write the CPU buffer (including the padding) to the file
        fwrite(h_buffer, sizeof(float), mtx->m * mtx->stride, f);
    }

    free(h_buffer);
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
            free((*results));
            *results = NULL;
        }
    }
}


float turboquant_mean_squared_error(turboquant_context_t *context, const vector_t *vec_1, const vector_t *vec_2) {
    if (vec_1 == NULL || vec_2 == NULL)
        return -1.0f;
    else if (vec_1->vector == NULL || vec_2->vector == NULL ||
            vec_1->n != vec_2->n)
        return -1.0f;
    
    int n = (int)vec_1->n;

    if (lin_alg_copy_vector(context->mse_buffer, vec_1) != SUCCESS)
        return -1.0f;

    // 2. Vector Subtraction: mse_buffer = (-1.0 * vec2) + mse_buffer
    if (lin_alg_sub_vectors(context->mse_buffer, vec_2) != SUCCESS)
        return -1.0f;

    // 3. L2 Norm: result = sqrt(sum(mse_buffer^2))
    float norm_result = 0.0f;

    if ((norm_result = lin_alg_l2(context->mse_buffer)) == -1.0f)
        return -1.0f;

    // 4. Square it and divide by N to get MSE
    float mse = (norm_result * norm_result) / (float)n;

    return mse;
}


/**
 * @param buffer: The destination byte array
 * @param dim_index: Which dimension we are saving (0, 1, 2...)
 * @param b: The bit width (e.g., 2)
 * @param value: The index of the centroid (e.g., 3)
 */
void turboquant_pack_dynamic(uint8_t *buffer, size_t dim_index, uint8_t b, uint8_t value) {
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

uint8_t turboquant_unpack_dynamic(const uint8_t *buffer, size_t dim_index, uint8_t b) {
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


__device__ static inline void turboquant_atomic_or_byte(uint8_t* base_addr, size_t pos, uint8_t val) {
    unsigned int* word_ptr = (unsigned int*)(base_addr + (pos & ~3));
    uint32_t shift = (pos & 3) * 8;
    atomicOr(word_ptr, (uint32_t)val << shift);
}

__global__ void turbo_quant_fused_kernel(
    const float* y,
    const float* centroids,
    int n_centroids,
    int bit_width,
    int dims,
    uint8_t* d_bstring)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dims) return;

    // 1. Centroid Search (Parallel)
    float val = y[i];
    float min_err = 1e10f;
    uint32_t best_idx = 0;

    for (int k = 0; k < n_centroids; k++) {
        float err = fabsf(val - centroids[k]);
        if (err < min_err) {
            min_err = err;
            best_idx = k;
        }
    }

    // 2. GPU-Safe Bit Packing (32-bit Word Alignment)
    size_t total_bits = (size_t)i * bit_width;
    size_t byte_pos = total_bits / 8;
    uint8_t bit_offset = total_bits % 8;

    uint8_t clean_value = (uint8_t)best_idx & ((1 << bit_width) - 1);
    uint16_t shifted_val = (uint16_t)clean_value << bit_offset;


    // Write the first byte
    turboquant_atomic_or_byte(d_bstring, byte_pos, (uint8_t)(shifted_val & 0xFF));

    // Handle the cross-byte boundary (The "Straddle")
    if (bit_offset + bit_width > 8) {
        turboquant_atomic_or_byte(d_bstring, byte_pos + 1, (uint8_t)(shifted_val >> 8));
    }
}

__global__ void turboquant_dequant_kernel(
    const uint8_t* d_bstring,
    const float* centroids,
    int bit_width,
    int dims,
    float* out_vector) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dims) return;

    // 1. Bit Unpacking (Mirroring your unpack_dynamic logic)
    size_t total_bits = (size_t)i * bit_width;
    size_t byte_pos = total_bits / 8;
    uint8_t bit_offset = total_bits % 8;

    // Read the "window" to handle values spanning two bytes
    uint16_t window = d_bstring[byte_pos];
    if (bit_offset + bit_width > 8) {
        window |= (uint16_t)d_bstring[byte_pos + 1] << 8;
    }

    // 2. Extract index and lookup centroid value
    uint8_t index = (uint8_t)((window >> bit_offset) & ((1 << bit_width) - 1));
    
    // Write directly to the GPU workspace
    out_vector[i] = centroids[index];
}

/* bstring buffer must be allocated with b-width * d prior to calling the function */
/* responsibility of the developer to zero out bstring */
uint8_t turboquant_mse_quantization(turboquant_context_t *context, const vector_t *x) {
    if (context == NULL || x == NULL)
        return QUANT_NULL;
    else if (x->vector == NULL || x->n == 0 || context->mse_buffer == NULL ||
            context->y == NULL || context->mse_quantizer == NULL) 
        return QUANT_NULL;
    else if (!context->is_init)
        return QUANT_UNINITIALIZED;

    if (lin_alg_copy_vector(context->mse_buffer, x) != SUCCESS)
        return QUANT_MSE_FAILED;

    /* L2 normalization on input vector */
    if (lin_alg_l2_normalize(context->mse_buffer) != MATH_OPS_SUCCESS)
        return QUANT_MSE_FAILED;

    /* Apply rotation matrix on resulting vector */
    if (lin_alg_dot_productmv(context->mse_quantizer->Π, context->mse_buffer, 
                context->y) != SUCCESS)
        return QUANT_MSE_FAILED;


    int threads = 128;
    int blocks = (context->mse_quantizer->dims + threads - 1) / threads;

    turbo_quant_fused_kernel<<<blocks, threads, 0, (cudaStream_t)context->compute_stream>>>(
        context->y->vector,
        context->mse_quantizer->d_centroids, // Flattened centroids from your Lloyd-Max
        context->mse_quantizer->book->n_centroids,
        context->mse_quantizer->bit_width,
        context->mse_quantizer->dims,
        context->d_bstring // GPU writes directly to your CPU memory here
    );

    return QUANT_SUCCESS;
}

vector_t* turboquant_mse_dequantization(turboquant_context_t *context) {
    if (context->h_bstring == NULL)
        return NULL;
    if (!context->is_init)
        return NULL;

    cudaStream_t stream = (cudaStream_t)context->compute_stream;
    int threads = 128; // Optimized for Llama head dimension
    int blocks = (context->mse_quantizer->dims + threads - 1) / threads;
    
    turboquant_dequant_kernel<<<blocks, threads, 0, stream>>>(
        context->d_bstring,
        context->mse_quantizer->d_centroids,
        context->mse_quantizer->bit_width,
        context->mse_quantizer->dims,
        context->mse_buffer->vector // Output to GPU buffer
    );

    lin_alg_transpose_matrix(context->mse_quantizer->Π);

    if (lin_alg_dot_productmv(context->mse_quantizer->Π, context->mse_buffer, context->y) != SUCCESS) {
        lin_alg_transpose_matrix(context->mse_quantizer->Π);
        return NULL;
    }

    lin_alg_transpose_matrix(context->mse_quantizer->Π);
    cudaStreamSynchronize(stream);

    return context->y;
}

__global__ void turboquant_qjl_sign_kernel(const float* y, uint32_t* d_qjl, int dims) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. Each thread checks the sign of its element
    // sign = 1 if negative, 0 otherwise
    int active = (i < dims);
    int sign = active && (y[i] < 0.0f);

    // 2. Warp-level collection: Pack 32 signs into one 32-bit integer
    uint32_t packed_signs = __ballot_sync(0xFFFFFFFF, sign);

    // 3. Only the "lane 0" (first thread) of each warp writes to the buffer
    if (active && (threadIdx.x % 32 == 0)) {
        d_qjl[i / 32] = packed_signs;
    }
}

/* init_turboquant must be called with bit_width - 1 passed as a parameter */
uint8_t turboquant_prod_quantization(turboquant_context_t *context, vector_t *x, quantization_result *results) {
    if (x == NULL || context == NULL || results == NULL)
        return QUANT_NULL;
    if (!context->is_init)
        return QUANT_UNINITIALIZED;

    cudaStream_t stream = (cudaStream_t)context->compute_stream;

    if (turboquant_mse_quantization(context, x) != QUANT_SUCCESS) 
        return QUANT_PROD_FAILED;

    vector_t* x_hat = turboquant_mse_dequantization(context);
    if (x_hat == NULL)
        return QUANT_PROD_FAILED;

    if (lin_alg_copy_vector(context->mse_buffer, x) != SUCCESS)
        return QUANT_PROD_FAILED;

    if (lin_alg_l2_normalize(context->mse_buffer) != MATH_OPS_SUCCESS)
        return QUANT_PROD_FAILED; 

    if (lin_alg_sub_vectors(context->mse_buffer, x_hat) != SUCCESS)
        return QUANT_PROD_FAILED;

    if (lin_alg_dot_productmv(context->mse_quantizer->S, context->mse_buffer, context->y) != SUCCESS)
        return QUANT_PROD_FAILED;

    /* 5. 1-Bit Sign Packing (QJL Phase) */
    int threads = 128;
    int blocks = (context->mse_quantizer->dims + threads - 1) / threads;

    // We use the context's pinned d_qjl buffer (ensure it's zeroed in context_init)
    turboquant_qjl_sign_kernel<<<blocks, threads, 0, stream>>>(
        context->y->vector,
        (uint32_t*)context->d_qjl,
        context->mse_quantizer->dims
    );

    // Link the result structure to the context's zero-copy buffers
    results->qjl = context->h_qjl;
    results->bstring = context->h_bstring;

    cudaStreamSynchronize(stream);
    results->residual_l2 = lin_alg_l2(context->mse_buffer);

    return QUANT_SUCCESS;
}

__global__ void turboquant_qjl_expand_kernel(const uint32_t* d_qjl, float* out, int dims) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dims) return;

    // Extract the bit corresponding to this thread from the 32-bit word
    uint32_t word = d_qjl[i / 32];
    uint32_t bit = (word >> (i % 32)) & 1;

    // Mapping: 1 (negative sign) -> -1.0f, 0 (positive sign) -> 1.0f
    out[i] = (bit == 1) ? -1.0f : 1.0f;
}

vector_t* turboquant_prod_dequantization(turboquant_context_t *context, const quantization_result *res) {
    if (res == NULL || context == NULL || !context->is_init) return NULL;

    cudaStream_t stream = (cudaStream_t)context->compute_stream;
    const size_t d = context->mse_quantizer->dims;

    /* Step 1: Reconstruct the primary MSE component */
    // Result is placed in context->y (rotated back to original space)
    if (turboquant_mse_dequantization(context) == NULL) return NULL;

    /* Step 2: Reconstruct the residual component signs */
    int threads = 128;
    int blocks = (d + threads - 1) / threads;
    
    // Expand 1-bit signs from res->qjl into context->mse_buffer
    // Note: res->qjl on CPU is mapped to context->d_qjl on GPU
    turboquant_qjl_expand_kernel<<<blocks, threads, 0, stream>>>(
        (uint32_t*)context->d_qjl, 
        context->mse_buffer->vector, 
        d
    );

    /* Step 3: Apply inverse rotation S^T to the residual signs */
    lin_alg_transpose_matrix(context->mse_quantizer->S);
    
    // We use context->mse_buffer as both input and workspace here 
    // (Assuming your dot_product wrapper handles non-overlapping buffers)
    // Residual = S^T * signs
    if (lin_alg_dot_productmv(context->mse_quantizer->S, context->mse_buffer, context->mse_buffer) != SUCCESS) {
        lin_alg_transpose_matrix(context->mse_quantizer->S); // Reset on failure
        return NULL;
    }
    lin_alg_transpose_matrix(context->mse_quantizer->S); // Reset state

    /* Step 4: Apply scaling factor */
    // Scale = (sqrt(pi/2) / d) * gamma
    float scale = (sqrtf(PI / 2.0f) / (float)d) * res->residual_l2;
    
    // Perform: mse_buffer = scale * mse_buffer (Using GPU scaling)
    lin_alg_scale_vector(context->mse_buffer, scale);

    /* Step 5: Final Sum (x_mse + x_residual) */
    // Perform: context->y = context->y + context->mse_buffer
    // This is essentially a GPU axpy operation (1.0 * mse_buffer + y)
    if (lin_alg_add_vectors(context->y, context->mse_buffer) != SUCCESS) {
        return NULL;
    }

    // Final Sync to ensure context->y is ready for the RAG engine
    cudaStreamSynchronize(stream);

    return context->y; 
}
