# TurboQuant SIMT (CUDA)

TurboQuantQuantization is a CUDA-based vector quantization prototype centered on TurboQuant-style MSE + residual (QJL sign) quantization. It targets NVIDIA GPUs using the SIMT (Single Instruction, Multiple Threads) execution model via the CUDA toolkit.

## Current scope
- CUDA linear algebra backend (cuBLAS + cuSOLVER + cuRAND)
- Codebook generation via Lloyd-Max over a scaled Beta distribution
- MSE quantization/dequantization and product quantization/dequantization flows
- Binary serialization for quantizer context (codebook + rotation matrices)

## Repository layout
- `include/` — public headers
- `src/codebook.c` — codebook generation (host-side)
- `src/lin_alg.cu` — CUDA linear algebra runtime + math ops
- `src/turboquant.cu` — quantization/dequantization CUDA kernels
- `src/initialize_context.c` — utility executable that precomputes and saves a context

## Build requirements
- CMake >= 3.20
- C11 compiler
- NVIDIA CUDA Toolkit (nvcc + libraries)
- cuBLAS, cuSOLVER, cuRAND (provided by CUDA Toolkit)

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

This builds:
- `libturboquant.so` — shared CUDA library
- `turboquant_context_factory` — executable for precomputing contexts
- `libturboquant_codebook.a` — host-only static library

### Non-CUDA local machine (host-only)

```bash
cmake -S . -B build -DTURBOQUANT_BUILD_CUDA=OFF
cmake --build build -j
```

This builds only:
- `libturboquant_codebook.a`

## Installation (shared library + headers)

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
    turboquant_context_t *ctx = NULL;
    const size_t dims = 128;
    const uint8_t mse_bit_width = 2; /* BIT_WIDTH - 1 when BIT_WIDTH is 3 */

    if (turboquant_init(&ctx, dims, mse_bit_width) != QUANT_SUCCESS) {
        fprintf(stderr, "turboquant_init failed\n");
        return 1;
    }

    if (turboquant_save(ctx, "turboquant_128_3bit.bin") != QUANT_SUCCESS) {
        fprintf(stderr, "turboquant_save failed\n");
        turboquant_context_destroy(&ctx);
        return 1;
    }

    turboquant_context_destroy(&ctx);
    return 0;
}
```

### 3) Load precomputed context

```c
if (turboquant_init_load(ctx, "turboquant_128_3bit.bin") != QUANT_SUCCESS) {
    fprintf(stderr, "Failed to load context\n");
    return 1;
}

/* ... quantize/dequantize without re-initializing matrices ... */

turboquant_context_destroy(&ctx);
```

## Generate a serialized TurboQuant context

After building:

```bash
./build/turboquant_context_factory
```

By default this writes:
- `turboquant_<DIMENSIONS>_<BIT_WIDTH>bit.bin`
  - Example with current `include/config.h`: `turboquant_128_3bit.bin`

## Execution

Run the context factory on a CUDA-enabled host:

```bash
./build/turboquant_context_factory
```

For applications linking against `libturboquant.so`, ensure the CUDA runtime libraries are in your `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```

## Performance notes

- Single-stream latency is dominated by kernel launch overhead (~50 μs) plus kernel execution time
- Best suited for batch sizes > 512 vectors to amortize launch cost
- Memory operations use zero-copy pinned host memory (`cudaHostAllocMapped`) for `bstring` and `qjl` buffers
- Stream synchronization happens automatically inside `turboquant_prod_dequantization`

## Notes

- Stream and CUDA math handle setup is managed internally by `lin_alg` runtime helpers
- The context factory uses `turboquant_init` / `turboquant_save` / `turboquant_context_destroy` (current API)
- All GPU memory is allocated via `cudaMalloc`; host-visible results use pinned zero-copy mappings
