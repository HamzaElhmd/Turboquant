#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "../../include/lin_alg.h"
#include "../../include/codebook.h"

/**
 * Verifies that the scaled Beta PDF is non-negative and properly 
 * normalized over the range [-1, 1].
 */
void test_codebook_scaled_beta_pdf() {
    size_t d = 1536; // target dimension[cite: 9]
    int N_samples = 1000; 
    float low = -1.0f, high = 1.0f;
    float dx = (high - low) / (float)N_samples;
    float area_sum = 0.0f;

    for (int j = 0; j <= N_samples; j++) {
        float x = low + (float)j * dx;
        float fx = codebook_scaled_beta_pdf(x, d); // Updated function name

        // Assert Non-negativity: PDF must never be negative[cite: 9]
        assert(fx >= 0.0f);

        // Trapezoidal integration weight[cite: 9]
        float w = (j == 0 || j == N_samples) ? 0.5f : 1.0f;
        area_sum += w * fx;
    }

    float total_area = area_sum * dx;

    // Assert Normalization: Integral must equal 1.0[cite: 9]
    assert(fabsf(total_area - 1.0f) < 1e-3f);
}

/**
 * Tests the Lloyd-Max quantization process, verifying ordering, 
 * midpoint boundaries (SIMD path), and distribution symmetry.
 */
void test_codebook_lloyd_max() {
    uint8_t bits = 3;         // 2^3 = 8 centroids (Triggers SIMD path)
    size_t dim = 1536;       
    uint16_t iterations = 200; 

    // Numerical method to generate centroids[cite: 7]
    codebook* cdb = codebook_lloyd_max(bits, dim, iterations); 
    assert(cdb != NULL);
    uint16_t n = (uint16_t)cdb->n_centroids;

    // 1. Verify SIMD Ordering: Centroids must be strictly increasing[cite: 9]
    for (int i = 0; i < n - 1; i++) {
        assert(cdb->centroids[i].value < cdb->centroids[i+1].value);
    }

    // 2. Verify SIMD Midpoint Property: Boundaries must be calculated as midpoints[cite: 8, 9]
    for (int i = 1; i < n - 1; i++) {
        float expected_boundary = (cdb->centroids[i].value + cdb->centroids[i+1].value) / 2.0f;
        // Verify current centroid's high_boundary matches the expected midpoint[cite: 8, 9]
        assert(fabsf(cdb->centroids[i].high_boundary - expected_boundary) < 1e-5f);
    }

    // 3. Verify Symmetry: Centroids should be symmetric around zero[cite: 9]
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += cdb->centroids[i].value;
    }
    assert(fabsf(sum) < 1e-3f);

    codebook_destroy(&cdb); // Updated destructor[cite: 7]
}

int main(int argc, char **argv) {
    srand(time(NULL));

    test_codebook_scaled_beta_pdf();
    fprintf(stdout, "\033[32mtest_codebook_scaled_beta_pdf(): success!\033[0m\n");

    test_codebook_lloyd_max();
    fprintf(stdout, "\033[32mtest_codebook_lloyd_max(): success!\033[0m\n");

    return 0;
}
