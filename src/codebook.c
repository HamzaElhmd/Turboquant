#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "../include/config.h"
#include "../include/codebook.h"

float codebook_scaled_beta_pdf(const float x, const size_t d) {
    if (x > 1.0f || x < -1.0f)
        return 0.0f;

    float a = 0.5f;
    float b = (d - 1) / 2.0f;

    float log_beta = lgammaf(a) + lgammaf(b) - lgammaf(a + b);

    float exponent = (d - 3) / 2.0f;
    float log_num = exponent * logf(1.0f - (x * x));

    float result = log_num - log_beta;

    return expf(result);
}

static void sort_centroids(codebook *cdb, uint16_t n) {
    int i, j;
    float key;
    
    for (i = 1; i < n; i++) {
        key = cdb->centroids[i].value; // The current element to be positioned
        j = i - 1;

        /* Move elements of arr[0..i-1] that are 
           greater than key, to one position ahead 
           of their current position */
        while (j >= 0 && cdb->centroids[j].value > key) {
            cdb->centroids[j + 1].value = cdb->centroids[j].value;
            j = j - 1;
        }
        cdb->centroids[j + 1].value = key;
    }
}

static uint16_t sort_compute_boundaries(codebook *cdb, const uint16_t n) {
    if (n == 0)
        return 1;

    sort_centroids(cdb, n);

    /* Calculate the boundaries */
    cdb->centroids[0].low_boundary = -1.0f;
    cdb->centroids[n-1].high_boundary = 1.0f;

    for (int i = 0; i < n; i++) {
        if (i == 0) {
            cdb->centroids[i].high_boundary = (cdb->centroids[i].value + 
                    cdb->centroids[i + 1].value) / 2.0f;
        } else if (i == n - 1) {
            cdb->centroids[i].low_boundary = (cdb->centroids[i].value + 
                    cdb->centroids[i - 1].value) / 2.0f;
        } else {
            cdb->centroids[i].high_boundary = (cdb->centroids[i].value + 
                    cdb->centroids[i + 1].value) / 2.0f;
            cdb->centroids[i].low_boundary = (cdb->centroids[i].value + 
                    cdb->centroids[i - 1].value) / 2.0f;
        }
    }

    return 0;
}

void codebook_destroy(codebook **cdb) {
    if (cdb) {
        if ((*cdb)->centroids)
            free((*cdb)->centroids);
        free(*cdb);
    }
    cdb = NULL;
}

codebook* codebook_init(const uint16_t n) {
     
    codebook *cdb = (codebook*) malloc(sizeof(codebook));
    if (cdb == NULL)
        return NULL;

    cdb->centroids = (centroid*) calloc(n, sizeof(centroid));
    if (cdb->centroids == NULL) {
        free(cdb);
        return NULL;
    }
    
    /* Initialize centroids with random values between -1 and 1 */
    /* Spread centroids evenly between -0.2 and 0.2
     * to ensure they start within the 'active' part of the PDF*/
    for (int i = 0; i < n; i++)
        cdb->centroids[i].value = -0.2f + (0.4f * (float)i / (n - 1));

        /* cdb->centroids[i].value = ((float) rand() / RAND_MAX) * 2.0f - 1.0f; */

    if (sort_compute_boundaries(cdb, n) != 0) {
        codebook_destroy(&cdb);
        return NULL;
    }

    cdb->n_centroids = n;

    return cdb;
}

#define CONVERGENCE_THRESHOLD 1e-6

codebook* codebook_lloyd_max(const uint8_t b, const size_t d, 
        const uint16_t max_iterations) {
    if (b == 0)
        return NULL;

    /* Based on the bit width get the number of centroids 2^b n */
    uint16_t n = (uint16_t) pow(2, b);

    /* Initialize random centroids */
    codebook *cdb = codebook_init(n);

    /* DEFINE N the number of samples to use in the trapezoid method */
    uint16_t j = 0, k = 0;
    float low_boundary, high_boundary, x, dx, fx, w,
          num_sum, denom_sum;
    
    while (k < max_iterations) {
        for (int i = 0; i < n; i++) {
            low_boundary = cdb->centroids[i].low_boundary;
            high_boundary = cdb->centroids[i].high_boundary;
            dx = (high_boundary - low_boundary) / (float) N;
            num_sum = 0.0f, denom_sum = 0.0f;

            for (int j = 0; j <= N; j++) {
                x = low_boundary + j * dx;
                w = (j == 0 || j == N) ? 0.5f: 1.0f;
                    
                fx = codebook_scaled_beta_pdf(x, d);
                num_sum += x * w * fx;
                denom_sum += w * fx;
            }

            if (denom_sum > 1e-20f) {
                cdb->centroids[i].value = num_sum / denom_sum;
            } else {
                // Nudge the centroid to the center of its interval
    // so it can 'find' the probability mass in the next iteration
                cdb->centroids[i].value = (low_boundary + high_boundary) / 2.0f;
            }
            
        }
        
        sort_compute_boundaries(cdb, n);
        k++;
    }

    return cdb;
}
