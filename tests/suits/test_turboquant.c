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

#ifndef PI
#define PI 3.14159265358979323846f
#endif

// External global state managed by turboquant.c[cite: 10]
extern turbo_quantizer *mse_quantizer;
extern uint8_t is_init;

/**
 * Helper to generate a random unit vector using the system vector_t
 */
vector_t* generate_random_unit_vector(size_t d) {
    vector_t *vec = lin_alg_create_vector(d);
    for (size_t i = 0; i < d; i++) {
        // Range [-1, 1][cite: 10]
        vec->vector[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
    lin_alg_l2_normalize(vec);
    return vec;
}

/**
 * Prints memory usage stats comparing raw vectors to quantized tripartite results
 */
void compare_memory_usage(size_t d, uint8_t b) {
    size_t raw_size = d * sizeof(float);
    
    // Calculate sizes based on the tripartite scheme (MSE + QJL + Residual)[cite: 10]
    // bstring stores b bits per dimension[cite: 10]
    size_t bstring_size = (d * b + 7) / 8; 
    // qjl stores 1 bit per dimension[cite: 10]
    size_t qjl_size = (d + 7) / 8;
    size_t residual_size = sizeof(float);
    size_t total_quant_size = bstring_size + qjl_size + residual_size;

    printf("\n--- Memory Usage Comparison (d=%zu, b=%u) ---\n", d, b);
    printf("Regular Vector (float[%zu]): %zu bytes\n", d, raw_size);
    printf("Quantized Result (Tripartite): %zu bytes\n", total_quant_size);
    printf("Compression Ratio: %.2fx\n", (float)raw_size / total_quant_size);
    
    // Read RSS (Resident Set Size) from /proc/self/status for Linux environments[cite: 10]
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

void test_init_cleanup() {
    size_t d = DIMENSIONS;
    uint8_t b = BIT_WIDTH;
    uint8_t err = turboquant_init(d, b);
    assert(err == QUANT_SUCCESS);
    turboquant_clean();
    fprintf(stdout, "\033[32mtest_init_cleanup(): success!\033[0m\n");
}

void test_mse_round_trip() {
    size_t d = DIMENSIONS;
    uint8_t b = BIT_WIDTH; 
    turboquant_init(d, b);

    vector_t *original = generate_random_unit_vector(d);
    size_t bytes = (d * b + 7) / 8;
    uint8_t *bstring = calloc(bytes, 1);

    assert(turboquant_mse_quantization(original, bstring) == QUANT_SUCCESS);
    vector_t *reconstructed = turboquant_mse_dequantization(bstring);
    assert(reconstructed != NULL);

    float mse = turboquant_mean_squared_error(original, reconstructed);
    printf("MSE Round-Trip Error: %f\n", mse);
    assert(mse < 0.25f); 

    lin_alg_free_vector(&original);
    lin_alg_free_vector(&reconstructed);
    free(bstring);
    turboquant_clean();
    fprintf(stdout, "\033[32mtest_mse_round_trip(): success!\033[0m\n");
}

void test_persistence() {
    size_t d = DIMENSIONS;
    uint8_t b = BIT_WIDTH;
    const char* filename = "turbo_context.bin";

    turboquant_init(d, b);
    float orig_Pi_0 = mse_quantizer->Π->matrix[0];

    assert(turboquant_save(filename) == QUANT_SUCCESS);
    turboquant_clean();

    assert(turboquant_init_load(filename) == QUANT_SUCCESS);
    assert(fabsf(mse_quantizer->Π->matrix[0] - orig_Pi_0) < 1e-6f);

    turboquant_clean();
    remove(filename);
    fprintf(stdout, "\033[32mtest_persistence(): success!\033[0m\n");
}

void test_prod_round_trip() {
    size_t d = DIMENSIONS;
    uint8_t b = BIT_WIDTH; 
    turboquant_init(d, b);

    vector_t *original = generate_random_unit_vector(d);
    quantization_result *res = turboquant_quantization_result_init();
    
    assert(turboquant_prod_quantization(original, res) == QUANT_SUCCESS);
    vector_t *reconstructed = turboquant_prod_dequantization(res);
    assert(reconstructed != NULL);

    float mse = turboquant_mean_squared_error(original, reconstructed);
    printf("Prod Round-Trip Error: %f\n", mse);
    assert(mse < 0.15f);

    lin_alg_free_vector(&original);
    lin_alg_free_vector(&reconstructed);
    turboquant_quantization_result_destroy(&res);
    turboquant_clean();
    fprintf(stdout, "\033[32mtest_prod_round_trip(): success!\033[0m\n");
}

int main() {
    srand(42); 

    test_init_cleanup();
    test_mse_round_trip();
    test_persistence();
    test_prod_round_trip();
    
    // Print memory stats for a common configuration[cite: 10]
    compare_memory_usage(DIMENSIONS, BIT_WIDTH);

    return 0;
}
