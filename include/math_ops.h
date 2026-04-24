#ifndef MATH_OPS
#define MATH_OPS

#include <stdint.h>
#include <stddef.h>

#define N 10000

typedef struct {
        float value;
        float low_boundary;
        float high_boundary;
} centroid;

typedef struct {
        float k;
        centroid *centroids;
} codebook;

void destroy_codebook(codebook **cdb);

codebook* init_codebook(const uint16_t n);

/* Calculate the L2 norm of input vectors to ensure length is 1,
 * projecting the vector on the hypersphere S^d-1 */
uint8_t l2_normalization(float *vec, const size_t d);

/* Generate a square nxn matrix where each row is random normal distribution values */
float** normal_distribution_random_matrix(const uint16_t n);

/* Apply QR decomposition a square matrix to ensure orthogonality and uniformity */
uint8_t qr_decomposition(float **matrix, const size_t n);

/* Generate a random Normal distribution matrix and QR decomposition, 
 * induces  Beta distribution on the coordinates */

/* Beta distribution probability density function (normalized on a hypersphere space S^d-1)*/
float scaled_beta_pdf(const float x, const size_t d);

/* Numerical method to get the centroids from the scaled Beta distribution */
codebook* lloyd_max(const uint8_t b, const size_t d, 
                const uint16_t max_iterations);

#endif
