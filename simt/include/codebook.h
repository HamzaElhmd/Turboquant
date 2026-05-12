#ifndef CODEBOOK_H
#define CODEBOOK_H

#include <stdint.h>
#include <stddef.h>
#include "../include/lin_alg.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
        float value;
        float low_boundary;
        float high_boundary;
} centroid;

typedef struct {
        size_t n_centroids;
        centroid *centroids;
} codebook;

void codebook_destroy(codebook **cdb);

codebook* codebook_init(const uint16_t n);

/* Generate a random Normal distribution matrix and QR decomposition, 
 * induces  Beta distribution on the coordinates */

/* Beta distribution probability density function (normalized on a hypersphere space S^d-1)*/
float codebook_scaled_beta_pdf(const float x, const size_t d);

/* Numerical method to get the centroids from the scaled Beta distribution */
codebook* codebook_lloyd_max(const uint8_t b, const size_t d, 
                const uint16_t max_iterations);

#ifdef __cplusplus
}
#endif

#endif
