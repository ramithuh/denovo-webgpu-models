# Shared WebGPU Core (`packages/core/`)

Planned home for shared, zero-dependency browser ES modules extracted across `models/*`:

- Sharded `.safetensors` streaming loader with Web Crypto SHA-256 verification and multipart tensor reconstruction
- Cache API & OPFS weight caching with graceful Safari Private Mode (`SecurityError` / `QuotaExceededError`) fallback
- Common WGSL compute kernels (packed 4×I8 `u32` matmul, FP16 `unpack2x16float` matmul, LayerNorm / RMSNorm, Firefox `wgpu`/Naga-safe `var<private>` helpers)
- Headless Chromium real-GPU test runner helper (explicitly rejecting software adapters)

Each engine under `models/<name>/` currently remains self-contained (`shared/`, `cpu/`, `webgpu/`) so `sync-upstream.mjs` can vendor any model independently into `denovo.run`.
