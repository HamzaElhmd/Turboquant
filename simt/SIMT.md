# SIMT Execution Model

**SIMT** (Single Instruction, Multiple Threads) is the parallelism model used by NVIDIA GPUs. In SIMT, a single instruction is executed simultaneously by many threads organized into warps (32 threads), with each thread operating on its own data element.

## How TurboQuant Uses SIMT

The `simt/` and `simt-multi/` variants offload quantization kernels to NVIDIA GPUs using CUDA. The core linear algebra and bit-packing operations are implemented as CUDA kernels:

- **cuBLAS** — GPU-accelerated matrix-vector products, vector copies, and norms
- **cuSOLVER** — QR decomposition for orthogonal rotation matrix generation
- **cuRAND** — GPU random normal distribution for matrix initialization
- **Custom CUDA kernels** — Fused centroid search + bit packing, QJL sign packing, and bit unpacking

### Key Kernels

| Kernel | Purpose | Parallelism |
|--------|---------|-------------|
| `turbo_quant_fused_kernel` | Per-dimension centroid search + bit packing | 1 thread per dimension |
| `turboquant_dequant_kernel` | Bit unpacking + centroid lookup | 1 thread per dimension |
| `turboquant_qjl_sign_kernel` | 1-bit sign packing via warp ballot | 1 thread per dimension, warp-level reduction |
| `turboquant_qjl_expand_kernel` | Sign bit expansion to ±1.0f | 1 thread per dimension |

## Why SIMT for Quantization?

GPUs excel at data-parallel workloads with high arithmetic intensity:
- Thousands of concurrent threads hide memory latency
- High memory bandwidth (HBM) sustains large batch throughput
- Dedicated matrix multiply and transcendental math units

SIMT is ideal for:
- Large batch sizes (hundreds to thousands of vectors)
- High-dimensional vectors where kernel launch overhead is amortized
- Data-center deployments with abundant GPU capacity

## SIMT vs SIMD

| Aspect | SIMT (CUDA) | SIMD (AVX2) |
|--------|-------------|-------------|
| Execution width | 32 threads (warp) | 8 floats (256-bit) |
| Memory | GPU HBM (300–1000 GB/s) | CPU DDR4/DDR5 (50–200 GB/s) |
| Launch overhead | Kernel launch (~50 μs) | Function call (~1 μs) |
| Batch sweet spot | 512–10K vectors | 1–256 vectors |
| PCIe transfer | Required for CPU→GPU data | None (host-native) |

For large batches, SIMT bandwidth and SM parallelism win. For small batches or low-latency queries, SIMD avoids kernel launch and PCIe overheads.

## Further Reading

- `README.md` — Build and execution instructions for this variant
