# TurboQuant 

TurboQuant is a CPU-native vector quantization library using Intel AVX2 SIMD instructions. It implements TurboQuant-style MSE + residual (QJL sign) quantization for high-dimensional vectors, optimized for modern x86-64 CPUs with AVX2 and FMA3 support.

For detailed implementation documentation, see [SIMD-AVX.md](SIMD-AVX.md).

## Current scope

- AVX2 linear algebra backend (aligned memory, 256-bit vectorized ops)
- Codebook generation via Lloyd-Max over a scaled Beta distribution
- MSE quantization/dequantization and product quantization/dequantization flows
- Binary serialization for quantizer context (codebook + rotation matrices)
- Intel SVML for vectorized transcendental functions (log, cos, sqrt)

## Repository layout

- `include/` — public headers
- `src/codebook.c` — codebook generation with SIMD bitonic sort
- `src/lin_alg.c` — AVX2 linear algebra runtime + math ops
- `src/turboquant.c` — quantization implementation
- `src/initialize_context.c` — utility executable that precomputes and saves a context

## Build requirements

- CMake >= 3.20
- Intel oneAPI compiler suite (icx/icpx)
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
- `libturboquant.so` — shared library with AVX2 kernels
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

## Generate a serialized TurboQuant context

After building:

```bash
./build/turboquant_context_factory
```

By default this writes:
- `turboquant_<DIMENSIONS>_<BIT_WIDTH>bit.bin`
  - Example with current `include/config.h`: `turboquant_1536_3bit.bin`

## Performance notes

- Single-core AVX2 throughput: ~50K–100K blocks/sec (128D vectors)
- Multi-core (OpenMP) can scale to 400K–800K+ blocks/sec on 8+ cores
- Optimal batch size for single-core: 64–256 vectors (L2 cache resident)
- See [SIMD-AVX.md](SIMD-AVX.md) for detailed benchmarks and hardware recommendations.

## Notes

- All memory is 32-byte aligned via `posix_memalign` for `vmovaps` loads/stores
- The context factory uses `turboquant_init` / `turboquant_save` / `turboquant_clean` (current API)
- Multi-threading is caller-managed; the library itself is single-threaded AVX2
