#ifndef HASH_MAP_H
#define HASH_MAP_H

#include <stdint.h>

#define MAP_MAX_SIZE 1024

typedef struct {
    char key[MAP_MAX_SIZE];
    char value[MAP_MAX_SIZE];
} map_t;

typedef struct {
    map_t *items;
    uint16_t n_items;
} hash_map_t;

hash_map_t* hash_map_init();

uint8_t hash_map_add(hash_map_t *hmap, const char *key, 
        const char *value);

uint8_t hash_map_del(hash_map_t *hmap, const char *key);

char** hash_map_get_keys(const hash_map_t * const hmap);

uint8_t hash_map_destroy(hash_map_t **hmap);

#endif
