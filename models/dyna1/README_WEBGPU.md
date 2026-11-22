# dyna1-webgpu

Browser-native inference for the primary published [Dyna-1](https://github.com/WaymentSteeleLab/Dyna-1) model, supporting **sequence-only**, **sequence + structure** (`dyna1.py --pdb ... --use_pdb_seq`), and **structure-only** (`dyna1.py --pdb ...`) modes using the official [gelnesr/Dyna-1](https://huggingface.co/gelnesr/Dyna-1) classifier, [ESM-3 small](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1) backbone, and ESM-3 VQ-VAE `StructureTokenEncoder`. This port follows the sharded safetensors, dependency-free runtime, and upstream intermediate-trace validation approach of `genie3-webgpu`.

## Exact inference architecture

Upstream commit: `814a453fd85305536b919901e41062b91d5c9d86`. The primary model is **ESM-3 small, not ESM-2 or a LoRA adapter**. `dyna1.pt` contains only the trained classifier. ESM-3 is frozen; no LoRA merge is needed.

| Component | Inference behavior |
| --- | --- |
| Input | Canonical amino acids (ESM vocabulary, **no BOS/EOS**, matching `dyna1.py`) and/or backbone 3D `(N, CA, C)` Å coordinates or discrete ESM-3 structure tokens (`0..4095`, `4096` mask) |
| Structure tokenizer *(optional, lazy-loaded)* | ESM-3 VQ-VAE `StructureTokenEncoder` (`esm3_structure_encoder_v0.pth`): Gram-Schmidt local frames (`eps=1e-10`), `K=16` CA kNN graph, relative positional embedding (`bins=32`), 2 geometric attention + SwiGLU transformer blocks (`d_model=1024`, `v_heads=128`, `ffn=4096`, chunked by `192` residues to stay under the 128 MB default WebGPU storage buffer limit), `pre_vq_proj` (`1024 → 128`), and 4096-entry Euclidean codebook argmin |
| Input encoder | 1536 channels; base `embedding.weight[seqTok]` holds `seq_embed[seqTok] + struct_embed[4096] + const_other_missing_tracks`; structure-conditioned modes add `structure_tokens_embed_delta[structTok] = struct_embed[structTok] - struct_embed[4096]`; structure-only mode uses `seqTok = 32` (`<mask>`), matching `ESM3.forward(sequence_tokens=None, structure_tokens=...)` |
| Backbone | First **23 of 48** ESM-3 blocks, 24 attention heads × 64 channels; full-channel Q/K LayerNorm followed by split-half RoPE; pre-LN attention; SwiGLU with width 4096 |
| Residual scaling | Divide each backbone attention/FFN update by `sqrt(48/36)`, retaining the original 48-layer scaling |
| Geometry | Upstream `ESM_model.forward` does not pass `structure_coords` to `ESM3.forward`, so block-0 `geom_attn` has an all-false frame mask and remains zero |
| Classifier input | `hidden_states[22]`, **before** ESM-3 final LayerNorm; later ESM blocks/output heads do not influence the prediction |
| Classifier | Outer LayerNorm (`eps=1e-12`), 12 pre-LN transformer encoder layers, 6 heads × 256 channels, ReLU FFN width 2048, outer LayerNorm, linear `1536 → 1` |
| Inner normalization | `eps=1e-5`; dropout disabled |
| Probability | `sigmoid(logit - log(0.05/0.95))`, exactly the upstream prior correction |

The main exported inference graph has **840,417,793 parameters**, and the optional additive `structure_encoder/` bundle adds **36,939,648 parameters** (`87.73 MB` across 6 shards `<= 25 MB`, weight relRMS `4.06e-10` vs original BF16 `esm3_structure_encoder_v0.pth`, giving **100% exact discrete structure tokens** across all tested proteins). Existing `int8/`, `fp16/`, and `fp32/` bundles remain 100% untouched and backwards-compatible; sequence-only calls incur zero extra downloads or compute. Outputs are exchange probabilities and classifier logits; logits are not calibrated physical exchange rates in inverse seconds.

## Export and reproducibility

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r tools/requirements-export.txt
.venv/bin/python tools/download_weights.py
.venv/bin/python tools/export_dyna1.py
.venv/bin/python tools/dump_oracle.py
```

Downloads are restricted to the two required official checkpoints, verified by their full SHA-256 digests in `tools/download_weights.py`. `.cache/`, `.venv/`, and `weights/` are ignored. `tools/requirements-export.txt` records the actual Python 3.11 environment (PyTorch 2.15.0, NumPy 1.26.4). Oracle execution is deterministic CPU FP32 with eight threads and the upstream PyTorch MHA fast path disabled.

| Bundle | Shards | Bytes (safetensors, excluding manifest) | Weight relRMS |
| --- | ---: | ---: | ---: |
| FP32 | 210 | 3,361,737,868 | 0 |
| FP16 mixed | 97 | 1,744,615,172 | 3.1717933106e-5 |
| INT8 | 43 | 843,626,504 | 9.8631742805e-3 |

All biases and normalization vectors remain F32. FP16 stores matrices in F16 **except classifier blocks 0 and 1**, which remain F32. This explicit accuracy exception adds 62.9 MB: rounding every matrix to F16 produced logit relRMS `7.540e-4` / `6.198e-4`, and probability relRMS `4.405e-4` / `7.392e-4` (Trp-cage / ubiquitin), missing the required `5e-4` threshold. Keeping these two blocks in F32 fixes this without changing the threshold. INT8 uses symmetric per-output-row quantization of every matrix, `scale=max(abs(row))/127` (clamped for zero rows), with F32 scales and F32 vectors.

The measured CPU PyTorch bundle round trips against the original upstream forward are saved in [`fixtures/precision_metrics.json`](fixtures/precision_metrics.json):

| Precision | Protein | Logit relRMS | Probability relRMS | Probability correlation |
| --- | --- | ---: | ---: | ---: |
| FP32 | both | 0 | 0 | 1 |
| FP16 mixed | Trp-cage (20 aa) | 1.2262124964e-4 | 7.1888376763e-5 | 0.9999999974 |
| FP16 mixed | Ubiquitin (76 aa) | 1.2635243835e-4 | 1.5110183822e-4 | 0.9999999908 |
| INT8 | Trp-cage | 5.4653095779e-3 | 3.4437745862e-3 | 0.9999272776 |
| INT8 | Ubiquitin | 5.0307040364e-3 | 5.7986857972e-3 | 0.9995089280 |

### Manifest

`dyna1-safetensors-v1` follows the Genie3 fields: `model`, `precision`, `source`, `total_params`, `num_tensors`, `total_bytes`, `config`, `shards` (file, byte count, SHA-256, logical/raw tensor counts), and `tensors`.

A normal tensor has `{key, shard, dtype, shape, row_start}`; quantized entries add `{quant: {scheme: "symmetric_per_row", axis: 0, scale_tensor, scale_dtype: "F32", scale_shape}}`. **Every scale tensor is in the same shard as its INT8 data.**

Unlike Genie3, several matrices individually exceed the cap. Those logical tensors have `{dtype, shape, parts: [...]}` where each part uses the same record and gives its starting row. The loader reconstructs contiguous rows without transposition. Actual serialized file sizes, including headers, are asserted to be **≤25,000,000 bytes**. Do not assume one logical tensor always occupies one shard.

### Oracle

`fixtures/oracle_dyna1.json` and `.bin` store little-endian F32/I32 values for Trp-cage `NLYIQWLKDGGPSSGRPPPS` and the complete 76-residue ubiquitin sequence. The traces come from the original local upstream ESM-3 and `model/transformer.py` modules with strict checkpoint loading. They include tokens, input embeddings, **all 23 retained backbone blocks**, the selected final embeddings, **all 12 classifier blocks**, logits, and adjusted probabilities. The binary is about 21 MB and is deliberately committed; model weights are not.

## Shared loader and CPU reference

The zero-dependency ES modules in `shared/` and `cpu/` import no Node built-ins. The shared loader accepts browser `fetch` or an injected `read(url) → ArrayBuffer`, validates shard length and SHA-256 where Web Crypto is available, decodes safetensors, and handles multipart tensors and colocated INT8 scales. `tokenize()` uppercases and removes whitespace, then rejects empty, noncanonical, or over-limit input rather than silently adding unknown tokens.

```js
import { createDyna1CPU } from 'dyna1-webgpu/cpu';
const engine = await createDyna1CPU({ baseUrl: '/weights/fp32' });
const result = engine.predict('NLYIQWLKDGGPSSGRPPPS', {
  returnIntermediates: true,
});
console.log(result.probs, result.logits, result.residues, result.timingMs);
engine.destroy();
```

The CPU reference stores weights and activations in `Float32Array`. JavaScript dot products use Number accumulation followed by FP32 output rounding, so this is a numerical reference rather than an emulation of a particular PyTorch GEMM reduction order. It implements ESM-3 SwiGLU and the classifier's **ReLU**, not GELU (neither live branch uses GELU).

`npm run test:cpu` passes both tests, with no skips. All 35 transformer block traces and final outputs satisfy `relRMS < 1e-4` on both inputs. Final probability relRMS: **2.2948908216e-7** (Trp-cage), **3.0438020853e-7** (ubiquitin). Exact tensor metrics and timings are in [`fixtures/cpu_metrics.json`](fixtures/cpu_metrics.json). The CPU implementation is for correctness; browser interactive inference should use WebGPU.

## WebGPU API

```js
import { createDyna1Engine } from 'dyna1-webgpu/webgpu';
const controller = new AbortController();
const engine = await createDyna1Engine({
  baseUrl: '/weights', // root containing fp32/, fp16/, int8/
  precision: 'int8',  // default; also 'fp16' or 'fp32'
  cache: 'cache',     // Cache API; 'opfs' or 'none' also supported
  useF16: true,       // optional shader-f16 storage loads when supported
  signal: controller.signal,
  onProgress: ({ phase, loaded, total }) => {
    if (phase === 'weights') console.log(loaded / total);
  },
});
try {
  const { probs, logits, residues, timingMs } = await engine.predict(
    'NLYIQWLKDGGPSSGRPPPS', { signal: controller.signal },
  );
  console.log(probs, logits, residues, timingMs);
} finally {
  engine.destroy();
}
```

A caller can supply `device`; the engine then owns its buffers but leaves the device alive. Otherwise it requests an adapter/device and destroys the device on teardown. `destroy()` is idempotent. Predictions on a destroyed engine, invalid input, overlapping predictions, failed downloads, corrupted shards, shader validation errors, and nonfinite outputs reject. Abort signals stop loading or CPU command preparation; already submitted GPU commands finish before prediction cancellation returns. Device loss is reported on subsequent predictions.

All neural-network operations execute as WGSL compute dispatches. CPU work is limited to tokenization, weight decoding/upload, command construction, and reading results. No ONNX runtime, WASM library, network inference service, or JavaScript model-forward fallback is used. F32 accumulation and activations are used for all storage modes:

- F32 matrices: direct F32 loads.
- F16 matrices, core WebGPU: packed `u32` plus `unpack2x16float`.
- F16 matrices, optional `shader-f16`: native `array<f16>` loads converted to F32; this does not round activations to F16.
- INT8 matrices: four signed bytes per `u32`, dequantized per row in the matrix multiplication shader.

The folded embedding lookup is decoded once during upload. Other quantized matrices stay packed on the GPU. Scratch buffers are reused between transformer blocks in command order; all per-prediction buffers are destroyed after readback, including on errors. `engine.memoryUsage()` reports live owned GPU buffers and bytes. `returnIntermediates: true` retains the oracle-named trace buffers for debugging; omit it in normal inference. Supported input bounds are 1–1024 canonical residues; complete-model parity and latency have been measured at 20 and 76 residues, not at every accepted length.

Cache API and OPFS caching preserve raw shards, verify manifest hashes on load, and support offline reuse after the first download. Use immutable/pinned Hugging Face revision URLs to avoid stale unversioned manifests. Cache storage is origin-specific and subject to browser quota. Clear Cache API data with `await caches.delete('dyna1-v1')`; clear OPFS data with `await (await navigator.storage.getDirectory()).removeEntry('dyna1-v1', { recursive: true })`. Neither cache is needed to execute inference.

## Tests and demo

```bash
export PATH="/home/ruh/.nvm/versions/node/v22.21.1/bin:$PATH"
npm test
npm run test:cpu
npm run test:gpu
```

Tests fail rather than skip if weights, fixtures, Chromium, or a hardware adapter are missing. `test:gpu` compiles the WGSL variants and checks full-model parity, warm latency, cancellation, and GPU buffer cleanup. `npm test` also verifies Cache API and OPFS hits without network access and imports both runtime modules inside a real module Web Worker. Core modules contain no Node imports; Node built-ins are confined to test and packaging scripts.

The browser harness binds an ephemeral loopback port. Set `CHROME_PATH` if needed. The machine used the ARM64 Chromium 1243 binary (`chrome-linux-arm64/chrome`), rather than the x86 `chrome-linux64` path. Hardware selection needed **both** `--use-angle=vulkan` and `--use-vulkan=native` in addition to the Genie3 flags:

```text
--headless=new --no-sandbox --disable-gpu-sandbox --ozone-platform=headless
--enable-unsafe-webgpu --ignore-gpu-blocklist --enable-features=Vulkan
--use-angle=vulkan --use-vulkan=native --disable-vulkan-surface
--enable-dawn-features=vulkan_enable_f16_on_nvidia
```

`DISPLAY`/`WAYLAND_DISPLAY` are unset for the child browser, and `VK_ICD_FILENAMES` points to the NVIDIA ICD when available. Software adapters are explicitly rejected by the numerical hardware test.

Serve the repository on an available port, for example:

```bash
.venv/bin/python -m http.server 8088 --bind 127.0.0.1
```

Open `http://127.0.0.1:8088/demo/`. The demo provides precision selection, progress, cancellation, a residue probability plot, CSV download, and explicit model release. Model loading is a substantial one-time download. HTTPS or localhost is required for WebGPU.

## Vendoring and Hugging Face hosting

`sync-upstream.mjs` copies only the runtime modules, license, and documentation and creates a private ESM package wrapper. It does not include model weights, Python environments, fixtures, or caches. Relative imports remain intact. A local copy/import smoke test was performed under `.cache/vendor-smoke`.

```bash
# Preview; add --write to perform the copy into your selected destination.
node sync-upstream.mjs --dest /absolute/path/to/vendor/dyna1
```

ProteinDesignSpace has not been modified. To integrate there, use its vendor destination and import the exported API from its worker, pointing `baseUrl` at the hosted bundle root.

```bash
# Verify all shard caps/checksums and generate weights/README.md and bundle-index.json.
.venv/bin/python tools/prepare_hf_upload.py
# After choosing your own destination repository and visibility:
# hf upload OWNER/dyna1-webgpu weights . --repo-type model
```

The generated Hugging Face model card documents precision exceptions, source models, output semantics, and the custom loader. Use `https://huggingface.co/OWNER/dyna1-webgpu/resolve/REVISION` as `baseUrl` after publication. The verified bundles and model card are local; no Hugging Face upload or git push has been performed. Original Dyna-1 source/license files remain in this checkout.

The parity harness uses a fresh Chromium process for each precision mode. A repeat run that swapped all four full bundles on one external device stalled between modes on this system; its results were not accepted. Repeated predictions and teardown are tested within each isolated mode. For ordinary use, retain one loaded engine and let it own its device. The core-mode tests request devices **without** the `shader-f16` feature; the native-mode test enables it explicitly.

## Verified hardware results

Measured on **NVIDIA GB10**, driver **580.159.03**, Chromium **153.0.8010.12** (Playwright build 1243), Linux ARM64. The adapter reported `vendor=nvidia` and `isFallbackAdapter=false`. All 16 shader variants compiled with zero Dawn validation errors. All retained FP32 block traces met `relRMS < 1e-4`; FP16 logits and probabilities met `5e-4`; INT8 logits and probabilities met `1e-2` with correlation above `0.99`.

| GPU mode | Protein | Logit relRMS | Probability relRMS | Probability correlation | Median warm ms |
| --- | --- | ---: | ---: | ---: | ---: |
| fp32-core | trp_cage | 8.0542350108e-07 | 5.2845468919e-07 | 1.0000000000 | 59.1 |
| fp32-core | ubiquitin | 5.3288469812e-07 | 6.4533871923e-07 | 1.0000000000 | 116.6 |
| fp16-core | trp_cage | 1.2267845209e-04 | 7.1949606916e-05 | 0.9999999974 | 43.4 |
| fp16-core | ubiquitin | 1.2630487809e-04 | 1.5109399494e-04 | 0.9999999908 | 89.6 |
| fp16-native | trp_cage | 1.2267845209e-04 | 7.1949606916e-05 | 0.9999999974 | 42.2 |
| fp16-native | ubiquitin | 1.2630487809e-04 | 1.5109399494e-04 | 0.9999999908 | 87.6 |
| int8-core | trp_cage | 5.4653328879e-03 | 3.4437629578e-03 | 0.9999272724 | 40.5 |
| int8-core | ubiquitin | 5.0307254710e-03 | 5.7986747592e-03 | 0.9995089002 | 85.5 |

Warm timings are medians of three complete `predict()` calls per sequence, after the traced first call. They include JavaScript command preparation and GPU readback, exclude weight loading and initial pipeline compilation, and were measured without an exclusive GPU reservation. Cold and individual warm measurements, per-tensor maxAbs/relRMS/correlation, feature flags, and live weight-buffer bytes are recorded in [`fixtures/gpu_metrics.json`](fixtures/gpu_metrics.json). INT8 live weight buffers occupy about 844 MB, FP16 mixed 1.745 GB, and FP32 3.362 GB; scratch allocations are additional and reused between blocks.

The final `npm test` suite covers nine tests: browser caches/Worker imports; CPU sequence-only oracle parity; CPU structure-conditioned parity; tokenizer and structure input validation; the standalone demo; full hardware GPU sequence-only parity/latency/lifecycle; full hardware GPU structure-encoder + `seq+struct` + `struct-only` parity/latency/chunking/lazy-load; and shader compilation (22 WGSL variants). All nine pass with no skips.

## Structure-conditioned inference (`seq+struct` & `struct-only`)

The additive `structure_encoder/` bundle (`87,731,896` bytes across 6 safetensors shards `<= 25 MB`, weight relRMS `4.06e-10`) is lazily fetched only when `structureCoords`, `backbone3Coords`, or `structureTokens` are passed to `predict()` or `encodeStructure()`, or eagerly via `await engine.preloadStructureEncoder()`.

```js
import { createDyna1Engine } from 'dyna1-webgpu/webgpu';

const engine = await createDyna1Engine({
  baseUrl: 'https://huggingface.co/ramithuh/dyna1-webgpu/resolve/main',
  precision: 'fp16', // or 'int8' / 'fp32'
  cache: 'cache',
  onProgress: ({ phase, loaded, total }) => {
    // phase is 'weights' during main load, 'structure-encoder' on lazy structure load
  },
});

// 1. Sequence + structure (matches `dyna1.py --pdb ... --use_pdb_seq`)
// structureCoords accepts flat Float32Array(L*9) [(N,CA,C)x(x,y,z) Å], { n, ca, c }, or [L][3][3]
// Missing backbone atoms may be NaN / Infinity / null (masked identically to ESM-3).
const both = await engine.predict(sequence, { structureCoords });
console.log(both.mode, both.structureTokens, both.probs, both.logits);

// 2. Structure-only (matches `dyna1.py --pdb ...` without --use_pdb_seq)
const structOnly = await engine.predict(null, { structureCoords });

// 3. Standalone structure tokenization or reusing precomputed tokens
const { structureTokens } = await engine.encodeStructure(structureCoords);
const reused = await engine.predict(sequence, { structureTokens, mode: 'seq+struct' });
```

### Verified structure-conditioned hardware results (NVIDIA GB10 WebGPU)

Discrete `StructureTokenEncoder` tokens match PyTorch upstream **386/386 (100.0%)** across Ubiquitin (`1UBQ`, 76 aa), Cyclophilin A (`1OCA`, 165 aa), YjbJ rigid control (`1RYK`, 69 aa), and `1UBQ_missing` (3 residues with `NaN` backbone atoms), plus **228/228 (100.0%)** on a multi-chunk `L=228` (`>192` chunk boundary) test. `pre_vq` relRMS is `4.6e-7 – 5.0e-7`.

| Protein | Upstream `seq-only` mean (min–max) | Upstream `seq+struct` mean (min–max) | Upstream `struct-only` mean (min–max) |
| --- | ---: | ---: | ---: |
| `1UBQ` (76 aa) | `0.5526` (`0.4041–0.8089`) | `0.2347` (`0.1081–0.5990`) | `0.2052` (`0.1098–0.6156`) |
| `1OCA` (165 aa) | `0.6626` (`0.4914–0.8501`) | `0.5350` (`0.2444–0.8817`) | `0.4724` (`0.2518–0.6946`) |
| `1RYK` (69 aa) | `0.5593` (`0.3590–0.7166`) | `0.2167` (`0.0842–0.6661`) | `0.2397` (`0.1123–0.5551`) |

| GPU mode | Protein | Tokens exact | `seq+struct` prob relRMS (`r`) | `struct-only` prob relRMS (`r`) | Encode ms | Total warm ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `fp32-core` | `1UBQ` | 76/76 | `2.33e-6` (`1.000000`) | `1.89e-6` (`1.000000`) | 82.3 | 194.0 |
| `fp32-core` | `1OCA` | 165/165 | `6.98e-7` (`1.000000`) | `9.01e-7` (`1.000000`) | 172.7 | 404.1 |
| `fp32-core` | `1RYK` | 69/69 | `1.74e-6` (`1.000000`) | `1.67e-6` (`1.000000`) | 74.2 | 183.7 |
| `fp32-core` | `1UBQ_missing` | 76/76 | `1.32e-6` (`1.000000`) | `1.51e-6` (`1.000000`) | 82.7 | 191.8 |
| `fp16-native` | `1UBQ` | 76/76 | `2.57e-4` (`1.000000`) | `4.00e-4` (`1.000000`) | 83.8 | 167.0 |
| `fp16-native` | `1OCA` | 165/165 | `6.61e-5` (`1.000000`) | `1.72e-4` (`1.000000`) | 178.2 | 350.1 |
| `fp16-native` | `1RYK` | 69/69 | `4.91e-4` (`1.000000`) | `2.17e-4` (`1.000000`) | 77.7 | 159.6 |
| `fp16-native` | `1UBQ_missing` | 76/76 | `2.29e-4` (`1.000000`) | `2.17e-4` (`1.000000`) | 84.1 | 167.3 |
| `int8-core` | `1UBQ` | 76/76 | `9.96e-2` (`0.979235`) | `6.93e-2` (`0.989429`) | 81.9 | 167.7 |
| `int8-core` | `1OCA` | 165/165 | `1.28e-2` (`0.999632`) | `2.15e-2` (`0.993291`) | 175.8 | 344.4 |
| `int8-core` | `1RYK` | 69/69 | `3.02e-2` (`0.998203`) | `2.81e-2` (`0.997810`) | 75.7 | 153.8 |
| `int8-core` | `1UBQ_missing` | 76/76 | `7.82e-3` (`0.999879`) | `8.51e-3` (`0.999961`) | 81.6 | 161.5 |

Full per-mode metrics are saved in [`fixtures/gpu_struct_metrics.json`](fixtures/gpu_struct_metrics.json) and [`fixtures/struct_precision_metrics.json`](fixtures/struct_precision_metrics.json). To re-export the structure encoder bundle and oracle locally:

```bash
.venv/bin/python tools/download_weights.py
.venv/bin/python tools/export_structure_encoder.py
.venv/bin/python tools/dump_struct_oracle.py
```
