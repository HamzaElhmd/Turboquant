# TurboQuant

TurboQuant is a CPU-native vector quantization library using Intel AVX2 SIMD instructions. It implements TurboQuant-style MSE + residual (QJL sign) quantization for high-dimensional vectors, optimized for modern x86-64 CPUs with AVX2 and FMA3 support.

For detailed implementation documentation, see [SIMD-AVX.md](SIMD-AVX.md).

## Current scope

- AVX2 linear algebra backend (aligned memory, 256-bit vectorized ops)
- Codebook generation via Lloyd-Max over a scaled Beta distribution
- MSE quantization/dequantization and product quantization/dequantization flows
- **Multi-threaded batch API** (OpenMP) with per-thread scratch buffers
- Binary serialization for quantizer context (codebook + rotation matrices)
- Intel SVML for vectorized transcendental functions (log, cos, sqrt)

## Repository layout

- `include/` — public headers
- `src/codebook.c` — codebook generation with SIMD bitonic sort
- `src/lin_alg.c` — AVX2 linear algebra runtime + math ops
- `src/turboquant.c` — quantization implementation (single-thread + batch API)
- `tests/suits/test_turboquant_mt.c` — multi-threaded batch test suite
- `src/initialize_context.c` — utility executable that precomputes and saves a context

## Build requirements

- CMake >= 3.20
- Intel oneAPI compiler suite (icx/icpx)
- OpenMP (bundled with Intel oneAPI)
- CPU with AVX2 + FMA3 support (Haswell/Excavator or newer)

### Check CPU support

```bash
grep -o 'avx2\|fma' /proc/cpuinfo | head -2
```

Both must appear.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

This builds:
- `libturboquant.so` — shared library with AVX2 + OpenMP batch kernels
- `test_turboquant` — single-threaded regression tests
- `test_turboquant_mt` — multi-threaded batch correctness + benchmark tests
- `turboquant_context_factory` — executable for precomputing contexts

## Installation (shared library + headers)

After building, install the shared library and headers to a prefix.

### Option A) User-local install (no sudo)

```bash
PREFIX="$HOME/.local"
install -d "$PREFIX/lib" "$PREFIX/include/turboquant"
install -m 755 build/libturboquant.so "$PREFIX/lib/"
install -m 644 include/*.h "$PREFIX/include/turboquant/"
```

Add this once (for runtime lookup):

```bash
export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH}"
```

### Option B) System-wide install

```bash
sudo install -d /usr/local/lib /usr/local/include/turboquant
sudo install -m 755 build/libturboquant.so /usr/local/lib/
sudo install -m 644 include/*.h /usr/local/include/turboquant/
sudo ldconfig
```

## Using the shared library once installed

### 1) Compile and link your application

```bash
PREFIX="$HOME/.local"  # or /usr/local
gcc app.c \
  -I"$PREFIX/include/turboquant" \
  -L"$PREFIX/lib" \
  -Wl,-rpath,"$PREFIX/lib" \
  -lturboquant \
  -o app
```

### 2) Minimal C usage example

```c
#include <stdio.h>
#include <stdint.h>
#include <turboquant.h>
#include <errors.h>

int main(void) {
    const size_t dims = 1536;
    const uint8_t mse_bit_width = 2;  /* BIT_WIDTH - 1 when BIT_WIDTH is 3 */

    if (turboquant_init(dims, mse_bit_width) != QUANT_SUCCESS) {
        fprintf(stderr, "turboquant_init failed\n");
        return 1;
    }

    if (turboquant_save("turboquant_1536_3bit.bin") != QUANT_SUCCESS) {
        fprintf(stderr, "turboquant_save failed\n");
        turboquant_clean();
        return 1;
    }

    turboquant_clean();
    return 0;
}
```

### 3) Load precomputed context

```c
if (turboquant_init_load("turboquant_1536_3bit.bin") != QUANT_SUCCESS) {
    fprintf(stderr, "Failed to load context\n");
    return 1;
}

/* ... quantize/dequantize without re-initializing matrices ... */

turboquant_clean();
```

### 4) Multi-threaded batch API

For high-throughput processing of many vectors, use the batch API which manages
per-thread scratch buffers internally and parallelizes over OpenMP threads:

```c
#include <stdlib.h>
#include <turboquant.h>

int main(void) {
    const size_t dims = 1536;
    const uint8_t bit_width = 2;
    const size_t n = 512;
    const size_t n_threads = 4;

    /* Allocate input vectors */
    vector_t **vecs = calloc(n, sizeof(vector_t *));
    for (size_t i = 0; i < n; i++)
        vecs[i] = lin_alg_create_vector(dims); /* fill ... */

    /* Initialize batch context with per-thread scratch buffers */
    turboquant_batch_ctx_t *ctx = NULL;
    if (turboquant_batch_init(&ctx, dims, bit_width, n_threads) != QUANT_SUCCESS) {
        fprintf(stderr, "batch init failed\n");
        return 1;
    }

    /* Quantize entire batch in parallel */
    quantization_result *results = calloc(n, sizeof(quantization_result));
    turboquant_batch_quantize(ctx, vecs, results, n);

    /* Dequantize entire batch in parallel */
    vector_t **reconstructed = turboquant_batch_dequantize(ctx, results, n);

    /* Save context for later reuse */
    turboquant_batch_save(ctx, "batch_ctx.bin");

    /* Cleanup */
    turboquant_batch_results_destroy(&reconstructed);
    for (size_t i = 0; i < n; i++) {
        free(results[i].bstring);
        free(results[i].qjl);
    }
    free(results);
    turboquant_batch_destroy(&ctx);
    return 0;
}
```

**Key properties:**
- Each thread gets its own isolated scratch buffers (no false sharing)
- `n_threads` controls the OpenMP team size; threads are pinned via `num_threads`
- Context is reusable across multiple batches without reallocation

## Generate a serialized TurboQuant context

After building:

```bash
./build/turboquant_context_factory
```

By default this writes:
- `turboquant_<DIMENSIONS>_<BIT_WIDTH>bit.bin`
  - Example with current `include/config.h`: `turboquant_1536_3bit.bin`

## Performance notes

- Single-core AVX2 throughput: ~378 vectors/sec (1536D, Ice Lake)
- Multi-core batch API: ~916 vectors/sec on 4 threads (2.4x speedup), ~1070 vectors/sec on 8 threads (2.8x speedup)
- Optimal batch size for single-core: 64–256 vectors (L2 cache resident)
- See [SIMD-AVX.md](SIMD-AVX.md) for detailed benchmarks and hardware recommendations.

## Notes

- All memory is 32-byte aligned via `posix_memalign` for `vmovaps` loads/stores
- The context factory uses `turboquant_init` / `turboquant_save` / `turboquant_clean` (current API)
- The batch API uses OpenMP with `num_threads` to bound the thread team to pre-allocated buffer count
- Single-threaded API (`turboquant_init` / `turboquant_prod_quantization`) remains available for low-latency single-vector workloads
