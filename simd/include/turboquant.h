#ifndef QUANTIZATION_H
#define QUANTIZATION_H

#include <complex.h>
#include <stdint.h>
#include <stddef.h>

#include "../include/codebook.h"
#include "../include/lin_alg.h"

/* TODO: USE config.h for MAX_ITERATIONS, N, and BIT_WIDTH, MAYBE
 * ALSO DIMENSION */

/* TODO: PACK THE HAAR ROTATION MATRIX Π and CODEBOOK c_ in a Quantizer matrix
 * To support multiple dimensions and bit widths */

/* TODO: RENAME FILE TO TURBOQUANT */

/* TODO: INCLUDE ERROR HANDLING BY CODE FOR FUNCTIONS THAT RETURN NULL */

/* TODO: PRECOMPUTE THE TRANSPOSE OF THE HAAR MATRIX IN THE QUANTIZATION CONTEXT STRUCT */

typedef struct {
        matrix_t *Π;
        matrix_t *t_Π;
        matrix_t *S;
        matrix_t *t_S;
        codebook *book;
        uint8_t bit_width;
        size_t dims;
} turbo_quantizer;

typedef struct {
        uint8_t *bstring;
        uint8_t *qjl;
        float residual_l2;
} quantization_result;


uint8_t turboquant_init(const size_t dim, const uint8_t bit_width);

void turboquant_clean();

uint8_t turboquant_init_load(const char *filename);

uint8_t turboquant_save(const char *filename);

turbo_quantizer* turboquant_quantizer_init(const size_t dims, const uint8_t bit_width);

void turboquant_quantizer_destroy(turbo_quantizer **quantizer);

quantization_result* turboquant_quantization_result_init();

void turboquant_quantization_result_destroy(quantization_result **results);

/* TODO: ADD SIMD TO THE LIN_ALG FUNCTIONS */

float turboquant_mean_squared_error(const vector_t *vec_1,
                const vector_t *vec_2);


void turboquant_pack_dynamic(uint8_t *buffer, size_t dim_index, uint8_t b, uint8_t value);

uint8_t turboquant_unpack_dynamic(const uint8_t *buffer, size_t dim_index, uint8_t b);

uint8_t turboquant_mse_quantization(const vector_t *x, uint8_t *bstring);

/* uint8_t quantized_john_lidenstrauss(const float *vec, const size_t d, 
                int8_t bit); */

vector_t* turboquant_mse_dequantization(const uint8_t *bstring);

uint8_t turboquant_prod_quantization(vector_t *x, quantization_result *results);

vector_t* turboquant_prod_dequantization(const quantization_result *res);


#endif
