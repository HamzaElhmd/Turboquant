#ifndef MATH_OPS
#define MATH_OPS

#include <stdint.h>
#include <stddef.h>

typedef struct {
        float value;
        float boundary;
} centroid;

typedef struct {
        float k;
        centroid *centroids;
} codebook;

/* Calculate the L2 norm of input vectors to ensure length is 1,
 * projecting the vector on the hypersphere S^d-1 */
uint8_t l2_normalization(float *vec, const size_t d);

/* Generate a square nxn matrix where each row is random normal distribution values */
float* normal_distribution_random(const uint16_t n);

/* Apply QR decomposition a square matrix to ensure orthogonality and uniformity */
uint8_t qr_decomposition(float **matrix, const size_t n);

/* Generate a random Normal distribution matrix and QR decomposition, 
 * induces  Beta distribution on the coordinates */
float** haar_rotation_matrix_init(uint16_t random_seed);

uint8_t haar_rotation_matrix_destroy(float **rotation_matrix);

/* Beta distribution probability density function (normalized on a hypersphere space S^d-1)*/
float scaled_beta_pdf(const float x, const size_t d);

float gamma_func(const float n);

/* Numerical method to get the centroids from the scaled Beta distribution */
codebook* lloyd_max(const uint8_t b);

#endif
