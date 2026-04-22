#include <stdlib.h>
#include <string.h>

#include "../include/hash_map.h"

#define INITIAL_N_MAPS 0xFFF

hash_map_t* hash_map_init() {
    hash_map_t *hmap = (hash_map_t*) malloc(sizeof(hash_map_t));
    if (hmap == NULL)
        return NULL;

    hmap->n_items = 0;
    hmap->items = (map_t*) calloc(INITIAL_N_MAPS, sizeof(map_t));
    if (hmap->items == NULL) {
        free(hmap);
        return NULL;
    }
    hmap->map_capacity = INITIAL_N_MAPS;

    return NULL;
}

void hash_map_destroy(hash_map_t **hmap) {
    if (hmap != NULL) {
        if ((*hmap) != NULL && (*hmap)->items != NULL)
            free((*hmap)->items);
        free((*hmap));
    }

    *hmap = NULL; 

    return;
}

uint16_t hash(const float key) {
    uint32_t bits;
    memcpy(&bits, &key, sizeof(bits));
    
    bits ^= bits >> 16;
    bits *= 0x7feb352d;
    bits ^= bits >> 15;
    bits *= 0x846ca68b;
    bits ^= bits >> 16;

    return bits & 0xFFF;
}

uint8_t hash_map_add (hash_map_t *hmap, const float key, const uint8_t value) {
    if (hmap == NULL)
        return ERR_HMAP_NULL;
    else if (hmap->items == NULL)
        return ERR_HMAP_UNINIT;
    
    uint16_t index = hash(key), i = index;

    while (i != index - 1) {
        if (i == hmap->map_capacity - 1) 
            i = 0;
        
        if (hmap->items[i].is_set == 0b0) {
            hmap->items[i].key = key;
            hmap->items[i].value = value;
            hmap->items[i].is_set= 0b1;
            hmap->n_items++;
            break;
        } else 
            i++;
    }

    return HMAP_SUCCESS;
}

uint8_t hash_map_del(hash_map_t *hmap, const float key) {
    if (hmap == NULL)
        return ERR_HMAP_NULL;
    else if (hmap->items == NULL)
        return ERR_HMAP_UNINIT;

    uint16_t index = hash(key);

    hmap->items[index].is_set = 0b0;
    hmap->n_items--;

    return HMAP_SUCCESS;
}


float* hash_map_get_keys(const hash_map_t * const hmap) {
    if (hmap == NULL)
        return NULL;
    else if (hmap->items == NULL)
        return NULL;

    float *keys = (float *) calloc(hmap->n_items, sizeof(float));
    if (keys == NULL)
        return NULL;

    for (int i = 0, j = 0; i < hmap->map_capacity; i++) {
        if (hmap->items[i].is_set == 0b1)
            keys[j++] = hmap->items[i].key;
    }

    return keys;
}
