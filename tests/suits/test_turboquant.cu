#include <complex.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <cuda_runtime.h>

#include "../../include/config.h"
#include "../../include/errors.h"
#include "../../include/lin_alg.h"
#include "../../include/turboquant.h"

/**
 * Helper to generate a random unit vector on the GPU
 */
vector_t* generate_random_unit_vector(size_t d) {
    vector_t *vec = lin_alg_create_vector(d);
    assert(vec != NULL);

    float *h_temp = (float*)malloc(d * sizeof(float));
    assert(h_temp != NULL);

    for (size_t i = 0; i < d; i++) {
        h_temp[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

    assert(cudaMemcpy(vec->vector, h_temp, d * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(lin_alg_l2_normalize(vec) == MATH_OPS_SUCCESS);

    free(h_temp);
    return vec;
}

void test_init_cleanup() {
    size_t d = DIMENSIONS;
    uint8_t b = BIT_WIDTH;
    turboquant_context_t *ctx = NULL;

    uint8_t err = turboquant_init(&ctx, d, b);
    assert(err == QUANT_SUCCESS);
    assert(ctx != NULL);
    assert(ctx->is_init == 1);

    turboquant_context_destroy(&ctx);
    assert(ctx == NULL);
    fprintf(stdout, "\033[32mtest_init_cleanup(): success!\033[0m\n");
}

void test_prod_round_trip() {
    size_t d = DIMENSIONS;
    uint8_t b = BIT_WIDTH;
    turboquant_context_t *ctx = NULL;

    assert(turboquant_init(&ctx, d, b) == QUANT_SUCCESS);
    assert(ctx != NULL);

    vector_t *original = generate_random_unit_vector(d);
    quantization_result *res = turboquant_quantization_result_init();
    assert(res != NULL);

    assert(turboquant_prod_quantization(ctx, original, res) == QUANT_SUCCESS);

    vector_t *reconstructed = turboquant_prod_dequantization(ctx, res);
    assert(reconstructed != NULL);

    float mse = turboquant_mean_squared_error(ctx, original, reconstructed);
    printf("Prod Round-Trip MSE: %f\n", mse);
    assert(mse >= 0.0f);
    assert(mse < 0.15f);

    lin_alg_free_vector(&original);
    turboquant_quantization_result_destroy(&res);
    turboquant_context_destroy(&ctx);
    fprintf(stdout, "\033[32mtest_prod_round_trip(): success!\033[0m\n");
}

void test_persistence() {
    size_t d = DIMENSIONS;
    uint8_t b = BIT_WIDTH;
    const char* filename = "turbo_context.bin";
    turboquant_context_t *ctx_save = NULL;
    turboquant_context_t *ctx_load = NULL;

    assert(turboquant_init(&ctx_save, d, b) == QUANT_SUCCESS);
    assert(ctx_save != NULL);

    float h_orig_val = 0.0f;
    assert(cudaMemcpy(&h_orig_val, ctx_save->mse_quantizer->Π->matrix, sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);

    assert(turboquant_save(ctx_save, filename) == QUANT_SUCCESS);
    turboquant_context_destroy(&ctx_save);

    assert(turboquant_init(&ctx_load, d, b) == QUANT_SUCCESS);
    assert(ctx_load != NULL);
    assert(turboquant_init_load(ctx_load, filename) == QUANT_SUCCESS);

    float h_load_val = 0.0f;
    assert(cudaMemcpy(&h_load_val, ctx_load->mse_quantizer->Π->matrix, sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(fabsf(h_load_val - h_orig_val) < 1e-6f);

    turboquant_context_destroy(&ctx_load);
    remove(filename);
    fprintf(stdout, "\033[32mtest_persistence(): success!\033[0m\n");
}

void test_mse_round_trip() {
    size_t d = DIMENSIONS;
    uint8_t b = BIT_WIDTH;
    turboquant_context_t *ctx = NULL;

    assert(turboquant_init(&ctx, d, b) == QUANT_SUCCESS);
    assert(ctx != NULL);

    vector_t *original = generate_random_unit_vector(d);
    assert(turboquant_mse_quantization(ctx, original) == QUANT_SUCCESS);

    vector_t *reconstructed = turboquant_mse_dequantization(ctx);
    assert(reconstructed != NULL);

    float mse = turboquant_mean_squared_error(ctx, original, reconstructed);
    printf("MSE Round-Trip Error: %f\n", mse);
    assert(mse >= 0.0f);
    assert(mse < 0.25f);

    lin_alg_free_vector(&original);
    turboquant_context_destroy(&ctx);
    fprintf(stdout, "\033[32mtest_mse_round_trip(): success!\033[0m\n");
}

/**
 * Prints memory usage stats comparing raw vectors to quantized tripartite results
 */
void compare_memory_usage(size_t d, uint8_t b) {
    size_t raw_size = d * sizeof(float);
    size_t bstring_size = (d * b + 31) / 32 * 4;
    size_t qjl_size = (d + 31) / 32 * 4;
    size_t residual_size = sizeof(float);
    size_t total_quant_size = bstring_size + qjl_size + residual_size;

    printf("\n--- Memory Usage Comparison (d=%zu, b=%u) ---\n", d, b);
    printf("Regular Vector (float[%zu]): %zu bytes\n", d, raw_size);
    printf("Quantized Result (Tripartite): %zu bytes\n", total_quant_size);
    printf("Compression Ratio: %.2fx\n", (float)raw_size / total_quant_size);

    FILE* fp = fopen("/proc/self/status", "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            if (strncmp(line, "VmRSS:", 6) == 0) {
                printf("Current Process Physical RAM (RSS): %s", line + 7);
                break;
            }
        }
        fclose(fp);
    }
}

int main() {
    srand(time(NULL));
    printf("Starting TurboQuant GPU Tests...\n");

    test_init_cleanup();
    test_mse_round_trip();
    test_prod_round_trip();
    test_persistence();
    compare_memory_usage(DIMENSIONS, BIT_WIDTH);

    printf("\nAll tests passed successfully.\n");
    return 0;
}
