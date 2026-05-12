#include <assert.h>
#include <stddef.h>

#include "../../include/hash_map.h"
#include "../../include/errors.h"

hash_map_t *hmap = NULL;

void test_hash_map_init() {
    hmap = hash_map_init();
    
    assert(hmap != NULL);
    assert(hmap->n_items == 0);
    assert(hmap->map_capacity == INITIAL_N_MAPS);
    assert(hmap->items != NULL);
}


void test_hash_map_destroy() {
    hash_map_destroy(&hmap);

    assert(hmap == NULL);
}


void test_hash() {
    float key_1 = 0.23f;
    float key_2 = 0.2304f;

    uint16_t index_1 = hash(key_1);
    uint16_t index_2 = hash(key_2);

    assert(index_1 != index_2);
}

void test_hash_map_add() {
    int err_code = hash_map_add(hmap, 0.23f, 0xAF);

    assert(err_code == HMAP_SUCCESS);
    assert(hmap->n_items == 1);

    uint16_t index = hash(0.23f);
    assert(hmap->items[index].is_set == 0b1);
    assert(hmap->items[index].key == 0.23f);
    assert(hmap->items[index].value == 0xAF);
}

void test_hash_map_del() {
    uint16_t err_code = hash_map_del(hmap, 0.23f);

    assert(err_code == HMAP_SUCCESS);
    assert(hmap->n_items == 0);
    
    uint16_t index = hash(0.23f);
    assert(hmap->items[index].is_set == 0b0);
}

#include <stdio.h>

int main (int argc, char **argv) {
    fprintf(stdout, "Running test suits...\n");

    test_hash_map_init();
    fprintf(stdout, "\033[32mtest_hash_map_init(): success!\033[0m\n");

    test_hash();
    fprintf(stdout, "\033[32mtest_hash_init(): success!\033[0m\n");

    test_hash_map_add();
    fprintf(stdout, "\033[32mtest_hash_map_add(): success!\033[0m\n");

    test_hash_map_del();
    fprintf(stdout, "\033[32mtest_hash_map_del(): success!\033[0m\n");

    test_hash_map_destroy();
    fprintf(stdout, "\033[32mtest_hash_map_destroy(): success!\033[0m\n");
    
    return 0;
}
