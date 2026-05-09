#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../include/turboquant.h"
#include "../include/config.h"
#include "../include/errors.h"

/**
 * PRODUCTION FACTORY SCRIPT
 * Purpose: Generates the stable rotation matrices and codebook 
 * for a 1536-dimensional vector database.
 */
int main(void) {
    // 1. Setup randomness seed
    srand((unsigned int)time(NULL));
    turboquant_context_t *context = NULL;

    printf("--- TurboQuant Context Factory ---\n");
    printf("Target Dimensions: %d\n", DIMENSIONS);
    printf("Target Bit-Width:   %d (using %d for MSE)\n", BIT_WIDTH, BIT_WIDTH - 1);

    // 2. Run the heavy initialization
    // We use BIT_WIDTH - 1 because Algorithm 2 requires one bit 
    // to be reserved for the QJL residual sign.
    uint8_t status = turboquant_init(&context, DIMENSIONS, BIT_WIDTH - 1);

    if (status != QUANT_SUCCESS) {
        fprintf(stderr, "Error: Failed to initialize TurboQuant context.\n");
        return 1;
    }

    printf("Initialization successful (Matrices generated and QR decomposed).\n");

    // 3. Serialize to disk
    char filename[64];
    snprintf(filename, sizeof(filename), "turboquant_%d_%dbit.bin", DIMENSIONS, BIT_WIDTH - 1);

    if (turboquant_save(context, filename) == QUANT_SUCCESS) {
        printf("\033[32mSuccess: Context saved to '%s'\033[0m\n", filename);
        printf("You can now load this file in your PoC to bypass training.\n");
    } else {
        fprintf(stderr, "Error: Failed to save context to disk.\n");
        turboquant_context_destroy(&context);
        return 1;
    }

    // 4. Cleanup and Exit
    turboquant_context_destroy(&context);
    return 0;
}
