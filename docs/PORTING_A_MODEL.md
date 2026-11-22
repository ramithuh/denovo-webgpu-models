# One-Shot WebGPU Model Porting Prompt (`denovo.run` Standard)

> [!TIP]
> **How to use**: Port **one model per agent invocation**. Fill in the `{{PLACEHOLDERS}}` at the top and run inside `/home/ruh/ramith/personal_projects/denovo-webgpu-models` targeting `models/{{MODEL_SLUG}}/`. Existing verified models (`models/esmc/`, `models/protpardelle/`, `models/ligandmpnn/`, `models/lasermpnn/`) live right next door as reference implementations.

---

## Prompt Template

```markdown
You are porting **ONE** open-weights protein model — **{{MODEL_NAME}}**
({{UPSTREAM_REPO_URL}}, pinned commit `{{UPSTREAM_COMMIT}}`, license `{{LICENSE}}`) —
into a complete, standalone, zero-dependency, browser-native **WebGPU** ES module package
at `models/{{MODEL_SLUG}}/` inside `/home/ruh/ramith/personal_projects/denovo-webgpu-models`.

Inspect sibling verified engines (`models/esmc/`, `models/protpardelle/`,
`models/ligandmpnn/`) to follow the exact same sharded-safetensors, WGSL, Web Worker,
and real-GPU Chromium verification architecture.

## Non-Negotiable Finish Condition (Read First!)
- **DO NOT stop at preprocessing, tokenization, archive inspection, symmetry helpers, or
  partial Milestone 1 prep.**
- You are **ONLY finished** when the **complete neural network forward pass** (every layer
  and output head through final logits / embeddings / coordinates) executes end-to-end in
  **real hardware WebGPU Chromium** and matches the pinned PyTorch checkpoint's **final
  output tensor** within `relRMS < 1e-4` (with zero skipped tests).
- Never push to Hugging Face or claim readiness until end-to-end GPU parity passes.

## Non-Negotiable Constraints
1. **Zero external runtime dependencies**: `shared/`, `cpu/`, and `webgpu/` must be pure
   ES modules with **zero** npm runtime dependencies and **zero** Node built-ins (Node APIs
   allowed only in `tools/` and `test/`).
2. **Pure WGSL WebGPU compute**: All neural-network ops run as WGSL compute shaders in
   WebGPU. No ONNX Runtime, no WASM ML fallback, no remote inference API. Also keep WGSL
   Firefox `wgpu`/Naga-compatible (use `var<private>` instead of dynamically indexed `const`
   arrays).
3. **Never commit upstream repos or large binaries**: Clone upstream Python code into
   gitignored `.cache/upstream/` and download checkpoints into gitignored `weights/`.
4. **No fake passes**: Tests must **fail, never skip**, if hardware WebGPU, weights, or
   fixtures are missing. Software WebGPU adapters (SwiftShader / llvmpipe) must be
   explicitly rejected.

---

## Milestone 1 — Architecture Audit, Mode Matrix & Reproducible Export

### 1.1 Exhaustive Mode & Input Coverage Matrix
Inspect the pinned upstream model in `.cache/upstream/` and enumerate every inference mode,
conditioning track, and edge case supported by `{{MODEL_NAME}}` (monomer vs multi-chain,
insertion codes, missing-density gaps, ligands/ions if supported, conditioning masks).
Document any intentionally pruned/deferred sub-models explicitly with upstream code proof.

### 1.2 Reproducible Weight Export (`tools/export_*.py`)
- Create an isolated Python environment (`uv venv .venv`) with pinned
  `tools/requirements-export.txt` and SHA-256 verified downloads (`tools/download_weights.py`).
- Export **FP32**, **FP16**, and **INT8** bundles to `weights/<variant>/<precision>/`
  using the `{{MODEL_SLUG}}-safetensors-v1` manifest:
  - Strict shard size cap: **≤ 25,000,000 bytes per `.safetensors` file (including header)**.
  - Support multipart row-split tensors (`parts: [...]`) if any matrix exceeds 25 MB.
  - Keep 1D biases, norms, and embeddings in **F32**; quantize 2D weights to F16 or
    symmetric per-output-row I8 with colocated `<name>.__scale` F32 companions in the
    **same shard**.
  - Record `model`, `precision`, `upstream_commit`, `total_params`, `num_tensors`,
    `total_bytes`, `config`, per-shard `byteLength` + `sha256`, and tensor-to-shard map.
- **Strict round-trip reload check**: Reload every exported bundle back into the upstream
  PyTorch model and measure export quantization error vs FP32 (`fixtures/export_metrics.json`).

### 1.3 Multi-Mode Intermediate-Trace Oracles (`tools/dump_oracle.py`)
Dump deterministic little-endian binary fixtures (`fixtures/oracle_*.json` + `.bin`, kept
in gitignored `weights/` or `.cache/fixtures/` if > 1 MB) from CPU FP32 upstream execution:
- Record inputs, preprocessing features/masks, **every major block hidden state**, and
  **final logits / coordinates / embeddings**.
- For stochastic sampling / diffusion: record exact decoding order / initial noise / uniform
  draws so JS & WGSL replay the exact upstream trajectory deterministically.
- Also dump identically-quantized FP16/INT8 PyTorch reference outputs so **GPU arithmetic
  parity** is tested separately from **weight-quantization error**.

---

## Milestone 2 — Shared Loader & Zero-Dependency CPU Reference (`shared/`, `cpu/`)
- `shared/`: Sharded safetensors loader (browser `fetch` or injected `read(url)`), SHA-256
  validation via Web Crypto, multipart tensor reconstruction, optional Cache API / OPFS
  with graceful Safari Private Mode fallback, plus pure-JS parser/tokenizer matching
  upstream bit-for-bit.
- `cpu/`: Complete FP32 CPU reference executing **all layers through final output**.
  `npm run test:cpu` must verify intermediate traces and **final output** with
  `relRMS < 1e-4`.

---

## Milestone 3 — Pure WGSL WebGPU Runtime & Hardware Verification (`webgpu/`)
- Export `create{{ModelPascal}}Engine({ baseUrl, precision, device?, cache?, useF16?, signal?, onProgress? })`:
  - Supports external or self-managed `GPUDevice`; idempotent `destroy()`; `memoryUsage()`.
  - F32 accumulation for all precisions (`fp32`, `fp16` via `unpack2x16float` + optional
    `shader-f16`, packed 4×I8 `u32` `int8`).
  - Reuses scratch buffers across blocks; destroys all transient GPU buffers in `finally`
    (including on error and `AbortSignal` cancellation).
  - Enforces `maxBufferSize` / `maxStorageBufferBindingSize` checks before dispatch.
- Run headless Chromium on real hardware WebGPU (`npm run test:gpu` & `npm test`):
  1. **End-to-end final output parity** across every mode (`relRMS < 1e-4` vs identically
     quantized PyTorch, plus FP16/INT8 quality vs FP32).
  2. **Buffer hygiene**: `0` leaked transient buffers after normal runs, errors, and aborts;
     `0` bytes after `destroy()`.
  3. **Module Web Worker + Cache/OPFS offline hit + Safari Private Mode fallback**.
  4. **Cold load + warm GPU latency benchmarks** on real hardware.

---

## Milestone 4 — Demo, Vendor Sync & Documentation
1. `demo/`: Minimal static browser demo (`index.html` + JS) with precision selector,
   progress bar, cancel, result download, and Release GPU button.
2. `sync-upstream.mjs`: Copies ONLY runtime modules (`shared/`, `webgpu/`, optional `cpu/`),
   `LICENSE`, and `README_WEBGPU.md` into `ProteinDesignSpace/web/static/vendor/{{MODEL_SLUG}}/`.
3. `README_WEBGPU.md`: Exact upstream commit, tensor/byte export table, full **Mode Matrix
   final-output parity table**, GPU device & warm timings, and proposed `denovo.run` port
   mapping.
```
