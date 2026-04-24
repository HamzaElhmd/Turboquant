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

#endif
