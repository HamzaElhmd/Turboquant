# TurboQuantQuantization
TurboQuantQuantization is a CUDA-based vector quantization prototype centered on TurboQuant-style MSE + residual (QJL sign) quantization.
This repository is intended to be built on CUDA-enabled environments such as Google Colab.

## Current scope
- CUDA linear algebra backend (cuBLAS + cuSOLVER + cuRAND)
- Codebook generation via Lloyd-Max over a scaled Beta distribution
- MSE quantization/dequantization and product quantization/dequantization flows
- Binary serialization for quantizer context (codebook + rotation matrices)

The old hash map module has been removed from this project because it is not used by the current pipeline.

## Repository layout
- `include/` — public headers
- `src/codebook.c` — codebook generation
- `src/lin_alg.cu` — CUDA linear algebra runtime + math ops
- `src/turboquant.cu` — quantization implementation
- `src/initialize_context.c` — utility executable that precomputes and saves a context

## Build requirements
- CMake >= 3.20
- C11 compiler
- For CUDA targets:
  - NVIDIA CUDA Toolkit (nvcc + libraries)
  - cuBLAS, cuSOLVER, cuRAND (provided by CUDA Toolkit)

## Build
### 1) CUDA build (Google Colab / CUDA host)
```bash
cmake -S . -B build
cmake --build build -j
```

This builds:
- `turboquant` (static library)
- `turboquant_context_factory` (executable)
- `turboquant_codebook` (host-only static library)

### 2) Non-CUDA local machine (configure-only or host-only codebook target)
```bash
cmake -S . -B build -DTURBOQUANT_BUILD_CUDA=OFF
cmake --build build -j
```

This builds only:
- `turboquant_codebook`

## Generate a serialized TurboQuant context
After a CUDA build:
```bash
./build/turboquant_context_factory
```

By default this writes:
- `turboquant_<DIMENSIONS>_<BIT_WIDTH-1>bit.bin`
  - Example with current `include/config.h`: `turboquant_128_3bit.bin`

## Notes
- Stream and CUDA math handle setup is managed internally by `lin_alg` runtime helpers.
- The context factory uses `turboquant_init` / `turboquant_save` / `turboquant_context_destroy` (current API).
