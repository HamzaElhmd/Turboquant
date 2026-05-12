# SIMD Execution Model

**SIMD** (Single Instruction, Multiple Data) is a processor-level parallelism technique where a single instruction operates on multiple data elements simultaneously within wide vector registers.

## How TurboQuant Uses SIMD

The `simd/` and `simd-multi/` variants exploit Intel AVX2 (256-bit wide) vector registers to process 8 `float32` values per instruction. This accelerates the core linear algebra kernels that dominate quantization and dequantization:

- **Dot products** — Fused multiply-add (`_mm256_fmadd_ps`) across 8 lanes
- **Vector scaling** — `_mm256_mul_ps` on 8 floats at once
- **Matrix-vector products** — Row-wise dot products, each unrolled for ILP
- **Random normal generation** — 8 parallel Box-Muller transforms via Intel SVML
- **Codebook sorting** — SIMD bitonic sort network for 8-element centroid lists

## Why SIMD for Quantization?

Vector quantization is naturally data-parallel:
- Each dimension is quantized independently (after rotation)
- Dot products, norms, and scales are element-wise reductions
- 128D–1536D vectors fit comfortably in CPU L1/L2 caches

SIMD eliminates the kernel-launch and PCIe-transfer overheads of GPU execution, making it ideal for:
- Low-latency single-vector queries
- Small-to-medium batches that fit in cache
- Environments without discrete GPUs

## SIMD vs SIMT

| Aspect | SIMD (AVX2) | SIMT (CUDA) |
|--------|-------------|-------------|
| Execution width | 8 floats (256-bit) | 32 threads (warp) |
| Memory | CPU DDR4/DDR5 (50–200 GB/s) | GPU HBM (300–1000 GB/s) |
| Launch overhead | Function call (~1 μs) | Kernel launch (~50 μs) |
| Batch sweet spot | 1–256 vectors | 512–10K vectors |
| Multi-device | CPU cores + SMT | Multiple GPUs + streams |

For small batches, SIMD latency wins. For very large batches, SIMT bandwidth wins.

## Further Reading

- `SIMD-AVX.md` — Detailed AVX2 intrinsic usage, kernel pseudocode, and benchmarks
- `README.md` — Build and execution instructions for this variant
