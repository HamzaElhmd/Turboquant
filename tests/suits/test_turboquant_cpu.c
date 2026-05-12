#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "../../include/config.h"
#include "../../include/errors.h"
#include "../../include/lin_alg.h"
#include "../../include/turboquant.h"
#include "../../include/math_ops.h"

// Access global state defined in turboquant.c
extern turbo_quantizer *mse_quantizer;
extern uint8_t is_init;

/**
 * Helper to generate a random unit vector on the CPU
 */
float* generate_random_unit_vector_cpu(size_t d) {
    float *vec = create_vec(d);
    for (size_t i = 0; i < d; i++) {
        vec[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
    l2_normalization(vec, d);
    return vec;
}

/**
 * Memory usage stats
 */
void compare_memory_usage(size_t d, uint8_t b) {
    size_t raw_size = d * sizeof(float);
    size_t bstring_size = (d * b + 7) / 8; 
    size_t qjl_size = (d + 7) / 8;
    size_t residual_size = sizeof(float);
    size_t total_quant_size = bstring_size + qjl_size + residual_size;

    printf("\n--- Memory Usage Comparison (d=%zu, b=%u) ---\n", d, b);
    printf("Regular Vector (float[%zu]): %zu bytes\n", d, raw_size);
    printf("Quantized Result (Tripartite): %zu bytes\n", total_quant_size);
    printf("Compression Ratio: %.2fx\n", (float)raw_size / total_quant_size);
}

void test_init_cleanup() {
    uint8_t err = init_turboquant(DIMENSIONS, BIT_WIDTH);
    assert(err == QUANT_SUCCESS);
    assert(is_init == 1);
    clean_turboquant();
    assert(is_init == 0);
    fprintf(stdout, "\033[32mtest_init_cleanup(): success!\033[0m\n");
}

void test_mse_round_trip() {
    size_t d = DIMENSIONS;
    uint8_t b = BIT_WIDTH; 
    init_turboquant(d, b);

    float *original = generate_random_unit_vector_cpu(d);
    size_t bytes = (d * b + 7) / 8;
    uint8_t *bstring = calloc(bytes, 1);

    // Serial signature: mse_quantization(x, bstring)
    assert(mse_quantization(original, bstring) == QUANT_SUCCESS);
    float *reconstructed = mse_dequantization(bstring);
    assert(reconstructed != NULL);

    float mse = mean_squared_error(original, reconstructed, d);
    printf("MSE Round-Trip Error: %f\n", mse);
    assert(mse < 0.25f); 

    free_vec(original);
    free_vec(reconstructed);
    free(bstring);
    clean_turboquant();
    fprintf(stdout, "\033[32mtest_mse_round_trip(): success!\033[0m\n");
}

void test_persistence() {
    size_t d = DIMENSIONS;
    uint8_t b = BIT_WIDTH;
    const char* filename = "turbo_serial.bin";

    init_turboquant(d, b);
    float orig_val = mse_quantizer->Π[0][0];

    assert(save_turboquant(filename) == QUANT_SUCCESS);
    clean_turboquant();

    assert(init_load_turboquant(filename) == QUANT_SUCCESS);
    assert(fabsf(mse_quantizer->Π[0][0] - orig_val) < 1e-6f);

    clean_turboquant();
    remove(filename);
    fprintf(stdout, "\033[32mtest_persistence(): success!\033[0m\n");
}

void test_prod_round_trip() {
    size_t d = DIMENSIONS;
    uint8_t b = BIT_WIDTH; 
    init_turboquant(d, b);

    float *original = generate_random_unit_vector_cpu(d);
    quantization_result *res = init_quantization_result();
    
    // Serial signature: prod_quantization(x, results)
    assert(prod_quantization(original, res) == QUANT_SUCCESS);
    float *reconstructed = prod_dequantization(res);
    assert(reconstructed != NULL);

    float mse = mean_squared_error(original, reconstructed, d);
    printf("Prod Round-Trip Error: %f\n", mse);
    assert(mse < 0.15f);

    free_vec(original);
    free_vec(reconstructed);
    destroy_quantization_result(&res);
    clean_turboquant();
    fprintf(stdout, "\033[32mtest_prod_round_trip(): success!\033[0m\n");
}

int main() {
    srand(42); 

    printf("Starting TurboQuant Serial (CPU) Tests...\n");
    test_init_cleanup();
    test_mse_round_trip();
    test_persistence();
    test_prod_round_trip();
    
    compare_memory_usage(DIMENSIONS, BIT_WIDTH);

    return 0;
}
