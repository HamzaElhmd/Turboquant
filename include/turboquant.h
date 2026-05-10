#ifndef QUANTIZATION_H
#define QUANTIZATION_H

#include <complex.h>
#include <stddef.h>
#include <stdint.h>
#include "codebook.h"

#ifdef __cplusplus
extern "C" {
#endif

/* TODO: USE config.h for MAX_ITERATIONS, N, and BIT_WIDTH, MAYBE
 * ALSO DIMENSION */

/* TODO: PACK THE HAAR ROTATION MATRIX Π and CODEBOOK c_ in a Quantizer matrix
 * To support multiple dimensions and bit widths */

/* TODO: RENAME FILE TO TURBOQUANT */

/* TODO: INCLUDE ERROR HANDLING BY CODE FOR FUNCTIONS THAT RETURN NULL */

/* TODO: PRECOMPUTE THE TRANSPOSE OF THE HAAR MATRIX IN THE QUANTIZATION CONTEXT STRUCT */
typedef struct {
        matrix_t *Π;
        matrix_t *S;
        codebook *book;
        float *d_centroids;
        uint8_t bit_width;
        size_t dims;
} turbo_quantizer;

typedef struct {
	uint8_t *bstring;
	uint8_t *qjl;
	float residual_l2;
} quantization_result;

typedef struct {
	turbo_quantizer *mse_quantizer;
	vector_t *mse_buffer;
	vector_t *y;

	uint8_t *h_bstring;
	uint8_t *d_bstring;
	size_t bstring_size;

	uint8_t *h_qjl;
	uint8_t *d_qjl;
	size_t qjl_size;

	void *compute_stream;
	uint8_t is_init;
} turboquant_context_t;

/* Multi-stream batch processing context */
typedef struct {
	turboquant_context_t **contexts;
	uint8_t n_streams;
	size_t dims;
	uint8_t bit_width;
	uint8_t is_init;
} turboquant_batch_context_t;

/* Batch result for multiple vectors */
typedef struct {
	quantization_result *results;
	uint8_t n_results;
} quantization_batch_result;

uint8_t turboquant_init(turboquant_context_t **context, const size_t dim, const uint8_t bit_width);

void turboquant_clean(turboquant_context_t *context);

void turboquant_context_destroy(turboquant_context_t **context);

uint8_t turboquant_init_load(turboquant_context_t *context, const char *filename);

uint8_t turboquant_save(turboquant_context_t *context, const char *filename);

turbo_quantizer* turboquant_quantizer_init(const size_t dims, const uint8_t bit_width);

void turboquant_quantizer_destroy(turbo_quantizer **quantizer);

quantization_result* turboquant_quantization_result_init();

void turboquant_quantization_result_destroy(quantization_result **results);

/* TODO: ADD SIMD TO THE LIN_ALG FUNCTIONS */

float turboquant_mean_squared_error(turboquant_context_t *context, const vector_t *vec_1,
                const vector_t *vec_2);

void turboquant_pack_dynamic(uint8_t *buffer, size_t dim_index, uint8_t b, uint8_t value);

uint8_t turboquant_unpack_dynamic(const uint8_t *buffer, size_t dim_index, uint8_t b);

uint8_t turboquant_mse_quantization(turboquant_context_t *context, const vector_t *x);

/* uint8_t quantized_john_lidenstrauss(const float *vec, const size_t d,
                int8_t bit); */

vector_t* turboquant_mse_dequantization(turboquant_context_t *context);

uint8_t turboquant_prod_quantization(turboquant_context_t *context, vector_t *x, quantization_result *results);

vector_t* turboquant_prod_dequantization(turboquant_context_t *context, const quantization_result *res);

/* Multi-stream batch processing functions */
uint8_t turboquant_batch_init(turboquant_batch_context_t **batch_ctx, const size_t dim, const uint8_t bit_width, const uint8_t n_streams);
void turboquant_batch_destroy(turboquant_batch_context_t **batch_ctx);
uint8_t turboquant_batch_init_load(turboquant_batch_context_t *batch_ctx, const char *filename);
uint8_t turboquant_batch_save(turboquant_batch_context_t *batch_ctx, const char *filename);

uint8_t turboquant_prod_quantization_batch(turboquant_batch_context_t *batch_ctx, vector_t **x_array, quantization_batch_result *batch_results, const uint8_t batch_size);
vector_t** turboquant_prod_dequantization_batch(turboquant_batch_context_t *batch_ctx, const quantization_batch_result *batch_results);

#ifdef __cplusplus
}
#endif

#endif
