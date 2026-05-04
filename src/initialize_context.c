#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <wchar.h>
#include "../include/turboquant.h"
#include "../include/config.h"
#include "../include/errors.h"

/**
 * PRODUCTION FACTORY SCRIPT
 * Purpose: Generates the stable rotation matrices and codebook 
 * for a 1536-dimensional vector database.
 */
int main() {
    // 1. Setup randomness seed
    srand((unsigned int)time(NULL));

    printf("--- TurboQuant Context Factory ---\n");
    printf("Target Dimensions: %d\n", DIMENSIONS);
    printf("Target Bit-Width:   %d (using %d for MSE)\n", BIT_WIDTH, BIT_WIDTH - 1);

    // 2. Run the heavy initialization
    // We use BIT_WIDTH - 1 because Algorithm 2 requires one bit 
    // to be reserved for the QJL residual sign.
    uint8_t status = turboquant_init(DIMENSIONS, BIT_WIDTH - 1);

    if (status != QUANT_SUCCESS) {
        fprintf(stderr, "Error: Failed to initialize TurboQuant context.\n");
        return 1;
    }

    printf("Initialization successful (Matrices generated and QR decomposed).\n");

    // 3. Serialize to disk
    const char *filename = "turboquant_1536_2bit.bin";
    if (turboquant_save(filename) == QUANT_SUCCESS) {
        printf("\033[32mSuccess: Context saved to '%s'\033[0m\n", filename);
        printf("You can now load this file in your PoC to bypass training.\n");
    } else {
        fprintf(stderr, "Error: Failed to save context to disk.\n");
        turboquant_clean();
        return 1;
    }

    // 4. Cleanup and Exit
    clean_turboquant();
    return 0;
}
