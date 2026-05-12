#ifndef ERRORS_H
#define ERRORS_H

typedef enum {SUCCESS=0, ERROR=-1} lin_alg_error;

typedef enum {MATH_OPS_SUCCESS = 0, 
    MATH_OPS_NULL = 1,
    MATH_OPS_EMPTY = 2,
    MATH_QR_FAILED = 3,
    MATH_L2_FAILED = 4
} math_ops_error;

typedef enum {
    HMAP_SUCCESS = 0,
    HMAP_NULL = 1,
    HMAP_UNINIT = 2
} hash_map_error;

typedef enum {
    QUANT_SUCCESS = 0,
    QUANT_NULL = 1,
    QUANT_MSE_FAILED = 2,
    QUANT_PROD_FAILED = 3,
    QUANT_INIT_FAILED = 4,
    QUANT_UNINITIALIZED = 5,
    DEQUANT_SUCCESS = 0,
    DEQUANT_NULL = 6,
    DEQUANT_MSE_FAILED = 7
} quantization_error;

#endif
