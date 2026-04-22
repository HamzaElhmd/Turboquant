#ifndef QUANTIZATION_PRIM_H
#define QUANTIZATION_PRIM_H

#include <stdint.h>
#include <stddef.h>

float mean_squared_error(const float *vec_1,
                const float *vec_2, const size_t d);

uint8_t mse_quantization(const float *vec, const size_t d, 
        uint8_t *bstring, size_t bit_width);

uint8_t quantized_john_lidenstrauss(const float *vec, const size_t d, 
                int8_t bit);

uint8_t prod_quantization(float *residual, const size_t d);

uint8_t mse_dequantization(const uint8_t *bstring, const size_t bit_width, 
        float *vec, const size_t d);

uint8_t prod_dequantization(const float *vec, const size_t d);


#endif
