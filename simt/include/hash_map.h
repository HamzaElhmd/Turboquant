#ifndef HASH_MAP_H
#define HASH_MAP_H

#include <complex.h>
#include <stdint.h>

#define INITIAL_N_MAPS 0x3F

/* Error MACROS */

/* A structure for a unit map key->value */
typedef struct {
    float key; /* Centroid value */
    uint8_t value; /* Requested bit */
    uint8_t is_set : 1;
} map_t;

typedef struct {
    map_t *items;
    uint16_t n_items;
    uint16_t map_capacity;
} hash_map_t;

/* Initialize a hash map ADT */
hash_map_t* hash_map_init();

uint16_t hash(const float key);

uint8_t hash_map_add(hash_map_t *hmap, const float key, 
        const uint8_t value);

uint8_t hash_map_del(hash_map_t *hmap, const float key);

float* hash_map_get_keys(const hash_map_t * const hmap);

void hash_map_destroy(hash_map_t **hmap);

#endif
