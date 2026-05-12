# TurboQuant

TurboQuant is a high-performance vector quantization library implementing TurboQuant-style MSE + residual (QJL sign) quantization. It supports four execution variants optimized for different hardware parallelism models and concurrency levels.

## Variants

| Directory | Parallelism | Target Hardware | Best For |
|-----------|-------------|-----------------|----------|
| `simd/` | Single-threaded SIMD (AVX2) | Modern x86-64 CPUs | Low-latency single-vector quantization, cache-resident batches |
| `simd-multi/` | Multi-threaded SIMD (OpenMP + AVX2) | Multi-core x86-64 CPUs | High-throughput batch processing on CPU |
| `simt/` | Single-stream SIMT (CUDA) | NVIDIA GPUs | GPU-accelerated single-vector quantization |
| `simt-multi/` | Multi-stream SIMT (CUDA) | NVIDIA GPUs | GPU-accelerated batch processing with concurrent streams |

## What is TurboQuant?

TurboQuant quantizes high-dimensional floating-point vectors into compact bitstrings using:
1. **MSE quantization** — Lloyd-Max codebook over a Beta-scaled distribution
2. **QJL residual sign encoding** — 1-bit signs of the rotated residual after MSE reconstruction
3. **Product quantization** — Combined MSE bitstring + QJL signs for near-lossless round-trip

## Repository Layout

```
.
simd/           # AVX2 single-threaded CPU implementation
simd-multi/     # AVX2 + OpenMP multi-threaded CPU implementation
simt/           # CUDA single-stream GPU implementation
simt-multi/     # CUDA multi-stream GPU implementation
```

Each subdirectory is a self-contained CMake project with its own headers, sources, build system, and documentation.

## Quick Start

Choose the variant matching your hardware:

```bash
# CPU with AVX2 (any modern Intel/AMD x86-64)
cd simd
cmake -S . -B build
cmake --build build -j

# CPU with AVX2 + many cores
cd simd-multi
cmake -S . -B build
cmake --build build -j

# NVIDIA GPU
cd simt
cmake -S . -B build
cmake --build build -j
```

See the README inside each directory for detailed installation, build, and execution instructions.

## Documentation per Variant

- `simd/README.md` — Build, install, and usage for AVX2 single-threaded
- `simd/SIMD.md` — Overview of the SIMD execution model
- `simd/SIMD-AVX.md` — Deep-dive into AVX2 implementation details

- `simd-multi/README.md` — Build, install, and usage for AVX2 multi-threaded
- `simd-multi/SIMD.md` — Overview of the SIMD execution model
- `simd-multi/SIMD-AVX.md` — Deep-dive into AVX2 + OpenMP implementation details

- `simt/README.md` — Build, install, and usage for CUDA single-stream
- `simt/SIMT.md` — Overview of the SIMT (GPU) execution model

- `simt-multi/README.md` — Build, install, and usage for CUDA multi-stream
- `simt-multi/SIMT.md` — Overview of the SIMT (GPU) execution model

## Performance at a Glance

| Variant | Batch Size | Throughput (1536D vectors) | Latency |
|---------|-----------|---------------------------|---------|
| `simd` | 1 | ~378 vec/s | ~2.6 ms |
| `simd-multi` | 512 | ~1070 vec/s (8 threads) | ~0.9 ms/vec |
| `simt` | 1 | Kernel-limited | ~50 μs + launch |
| `simt-multi` | Large | SM-saturation dependent | Amortized |

**Rule of thumb**: Use `simd` for low-latency single queries, `simd-multi` for CPU batch jobs, `simt` for GPU offload with stream parallelism, and `simt-multi` for maximum GPU throughput with concurrent kernels.

## Common API Pattern

All variants share the same conceptual API:

1. `turboquant_init(ctx, dims, bit_width)` — initialize quantizer state
2. `turboquant_prod_quantization(ctx, vector, result)` — quantize
3. `turboquant_prod_dequantization(ctx, result)` — dequantize
4. `turboquant_save(ctx, filename)` / `turboquant_init_load(ctx, filename)` — persist state
5. `turboquant_clean(ctx)` — release resources

Batch variants (`simd-multi`, `simt-multi`) expose additional `turboquant_batch_*` functions for parallel multi-vector processing.

## License

See individual source files for license headers.
