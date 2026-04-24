#ifndef QUANTIZATION_H
#define QUANTIZATION_H

#include <complex.h>
#include <stdint.h>
#include <stddef.h>

#include "../include/math_ops.h"

/* TODO: USE config.h for MAX_ITERATIONS, N, and BIT_WIDTH, MAYBE
 * ALSO DIMENSION */

/* TODO: PACK THE HAAR ROTATION MATRIX Π and CODEBOOK c_ in a Quantizer matrix
 * To support multiple dimensions and bit widths */

/* TODO: RENAME FILE TO TURBOQUANT */

/* TODO: INCLUDE ERROR HANDLING BY CODE FOR FUNCTIONS THAT RETURN NULL */

/* TODO: PRECOMPUTE THE TRANSPOSE OF THE HAAR MATRIX IN THE QUANTIZATION CONTEXT STRUCT */
typedef struct {
        float **Π;
        float **t_Π;
        float **S;
        float **t_S;
        codebook *book;
        uint8_t bit_width;
        size_t dims;
} turbo_quantizer;

typedef struct {
        uint8_t *bstring;
        uint8_t *qjl;
        float residual_l2;
} quantization_result;


uint8_t init_turboquant(const size_t dim, const uint8_t bit_width);

void clean_turboquant();

uint8_t init_load_turboquant(const char *filename);

uint8_t save_turboquant(const char *filename);

turbo_quantizer* init_quantizer(const size_t dims, const uint8_t bit_width);

void destroy_quantizer(turbo_quantizer **quantizer);

quantization_result* init_quantization_result();

void destroy_quantization_result(quantization_result **results);

/* TODO: ADD SIMD TO THE LIN_ALG FUNCTIONS */

float mean_squared_error(const float *vec_1,
                const float *vec_2, const size_t d);


void pack_dynamic(uint8_t *buffer, size_t dim_index, uint8_t b, uint8_t value);

uint8_t unpack_dynamic(const uint8_t *buffer, size_t dim_index, uint8_t b);

uint8_t mse_quantization(const float *x, uint8_t *bstring);

/* uint8_t quantized_john_lidenstrauss(const float *vec, const size_t d, 
                int8_t bit); */

float* mse_dequantization(const uint8_t *bstring);

uint8_t prod_quantization(float *x, quantization_result *results);


float* prod_dequantization(const quantization_result *res);


#endif
