# EvoDiff-Seq 38M in WebGPU

A complete, standalone port of **OA_DM_38M**: token embedding, input projection, all 16 ByteNet residual blocks, final LayerNorm, 31-way decoder, and order-agnostic iterative sampling. Runtime files are plain ES modules with **no npm dependencies, Node built-ins, WASM, Python, or external inference runtime**. All neural operations in the WebGPU engine execute WGSL compute shaders. The separate CPU implementation is a reference backend.

## Provenance and architecture audit

- EvoDiff: [`33206e99446f799ec11cf6e57d66ffaec837be91`](https://github.com/microsoft/evodiff/tree/33206e99446f799ec11cf6e57d66ffaec837be91), MIT.
- Protein sequence models: [`af695772c4a1c056d930c95ec7e6428aa042f5cd`](https://github.com/microsoft/protein-sequence-models/tree/af695772c4a1c056d930c95ec7e6428aa042f5cd), MIT.
- Checkpoint: Alamdari et al., [`oaar-38M.tar`, Zenodo 8045076](https://zenodo.org/records/8045076), **CC BY 4.0**. Archive: **454,881,175 bytes**; MD5 `7d8959d06cc4373c220413ef00f99fab`; SHA-256 `cad6774580b700f563bdc63227c336351e8c0c7d0a89014122aa2dce4f768aa0`.
- Model state: **199 tensors, 37,890,327 parameters**, all originally FP32. Optimizer state is excluded. Exported weights retain attribution and describe the numerical conversions.

The source of truth is the pinned [38M config](https://github.com/microsoft/evodiff/blob/33206e99446f799ec11cf6e57d66ffaec837be91/config/config38M.json), [checkpoint constructor](https://github.com/microsoft/evodiff/blob/33206e99446f799ec11cf6e57d66ffaec837be91/evodiff/pretrained.py), [ByteNetTime](https://github.com/microsoft/evodiff/blob/33206e99446f799ec11cf6e57d66ffaec837be91/evodiff/model.py), and [ByteNetBlock / MaskedConv1d](https://github.com/microsoft/protein-sequence-models/blob/af695772c4a1c056d930c95ec7e6428aa042f5cd/sequence_models/convolutional.py).

| Stage | Exact behavior |
|---|---|
| Tokenizer | `ACDEFGHIKLMNPQRSTVWYBZXJOU-*#@!`; case-sensitive; no boundaries inserted |
| Special IDs | GAP 26, STOP 27, MASK 28, START 29, PAD 30 |
| Embedding | `[31,8]`, upstream `padding_idx=28`; inference gathers the stored row without altering it |
| Input projection | Conv1D `[1024,8,1]` + bias |
| Dilation schedule | `1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128` |
| Each block | LN(1024) → GELU → 1×1 Conv 1024→512 → LN(512) → GELU → dilated width-5 Conv 512→512 → LN(512) → GELU → 1×1 Conv 512→1024 → residual addition |
| Convolution | Cross-correlation, symmetric zero padding `2*dilation`, no causal masking; weights stored as `[out,in,kernel]` |
| Optional input mask | Multiplies the activated 512-channel input immediately before the width-5 convolution, at every block; it does not mask residual paths or infer padding automatically |
| LayerNorm | Per-position population variance, epsilon `1e-5`, learned scale and bias |
| Activation | Upstream `nn.GELU(approximate='none')`; WGSL/JS erf approximation absolute error below `1.5e-7`, verified through final logits |
| Final stages | LN(1024) → Conv1D `[31,1024,1]` + bias → row-major `[L,31]` logits |
| Disabled features | No timestep encoding (`timesteps=None`), dropout, tied decoder, or causal convolution |

The sampler follows the pinned [unconditional](https://github.com/microsoft/evodiff/blob/33206e99446f799ec11cf6e57d66ffaec837be91/evodiff/generate.py) and [conditional](https://github.com/microsoft/evodiff/blob/33206e99446f799ec11cf6e57d66ffaec837be91/evodiff/conditional_generation.py) OA-DM path: visit each masked position once in random order, rerun the entire network, softmax the selected position, and unmask it. **Upstream uses `len(MSA_AAS)-6 = 21` categories, including `B`**; this port preserves that behavior. Fixed motif positions remain unchanged. Replay replaces framework-specific multinomial RNG with recorded inverse-CDF draws and FP64 JS/Python softmax; native PyTorch RNG seeds are not interchangeable with the JS seed.

## Quick start

Serve this directory over HTTPS or localhost. Load one precision from a base URL that contains `fp32/`, `fp16/`, and `int8/`:

```js
import { createEvoDiffEngine } from './webgpu/index.js';

const engine = await createEvoDiffEngine({
  baseUrl: new URL('./weights/oa_dm_38m', location.href).href,
  precision: 'int8',
  cache: 'cache',
  onProgress: event => console.log(event),
});
try {
  const { logits, tokens } = await engine.forward('MSK#EELFTGVVPILVELDGDVNGH#FSVSGE');
  console.log(logits.length === tokens.length * 31);
  console.log((await engine.generate({ length: 24, seed: 42 })).sequence);
  console.log((await engine.generate({ fixedSequence: '####ACDE####', seed: 42 })).sequence);
} finally {
  engine.destroy();
}
```

`npm run serve` opens the static demo at `http://127.0.0.1:8080/demo/`. It provides unconditional/motif modes, precision and seed selectors, progress, cancellation, FASTA download, and GPU release.

### API contract

`createEvoDiffEngine({ baseUrl, precision='int8', device, cache='cache', useF16=false, read, signal, onProgress })` resolves after weights and pipelines are ready.

- `device`: optional external GPUDevice. The engine destroys a device only when it created it. Concurrent calls on one engine reject with `Engine busy`; use separate engines for concurrency.
- `useF16`: request/require `shader-f16` and use native F16 storage for FP16 matrix weights. With the default `false`, FP16 is unpacked from packed u32 storage. **All GPU accumulation and activations remain F32 in either case.** INT8 is packed four values per u32 and dequantized by its F32 output-channel scale in WGSL.
- `read(url, {signal})`: optional asynchronous reader returning an ArrayBuffer. Injected reads bypass browser caches, but shard hashes are always checked. No Node APIs are imported by the runtime.
- `cache`: `'none'`, `'cache'` (Cache API), or `'opfs'`. Storage security/quota errors fall back to network and emit `cache-warning`; cached bytes still require size/SHA-256 checks. Offline use requires the manifest and every shard to have been cached. Relative browser URLs should be resolved to absolute URLs for stable cache keys.
- `forward(sequenceOrTokens, { returnIntermediates=false, inputMask, signal })` returns `{logits: Float32Array, tokens: Int32Array, timingMs, intermediates?}`. `inputMask`, when supplied, contains one 0/1 per position and follows upstream masking semantics. Intermediates contain `embedding`, `projection`, `block.0` through `block.15`, `final_norm`, and `logits`. Arrays are flattened `[L,C]`; tokens use the full 31-entry alphabet.
- `generate({ length, fixedSequence, maskPositions, seed=0, order, uniforms, signal, onProgress })` returns `{sequence, tokens, steps, timingMs}`. Without a fixed sequence it starts from all MASK tokens. With a fixed sequence, `#` and the explicit zero-based `maskPositions` are regenerated. Other positions are preserved. `order` must be an exact permutation of all masked positions; `uniforms` must have one number in `[0,1)` per step. Otherwise a Mulberry32 stream and Fisher–Yates permutation are derived from the integer seed. Each step includes `position`, `uniform`, `token`, and full `[L,31]` logits.
- `memoryUsage()` reports tracked GPU allocations: weight/scratch/transient bytes, transient buffer count, total bytes, and buffer count. Driver pipeline memory and JavaScript arrays are not GPUBuffer allocations. Four reusable F32 scratch buffers, tokens, mask, and uniforms remain allocated until growth or destruction. Readback buffers are destroyed in `finally` after success, errors, and aborts.
- `destroy()` is idempotent. All tracked GPU buffers reach zero bytes. Abort cannot interrupt a submitted GPU command buffer; the engine waits for completion, rejects, and cleans up. Sampling yields between residues so DOM/Worker cancellation can run.

Buffer allocation and dispatch sizes are checked against device limits. The runtime supports one variable-length sequence per call. Explicit fallback/software adapter identification is rejected. Missing WebGPU, weights, fixtures, and unsupported requested features produce errors, never successful skips.

The CPU API is `createEvoDiffCPU({baseUrl, precision, read, cache, signal, onProgress})` from `cpu/index.js`, with matching `forward` and deterministic `generate`. Weights and activations use Float32Array; JS scalar arithmetic is used for the reference reductions. CPU forward is synchronous; use a Worker for responsive CPU execution.

## Reproduce the export and oracles

```sh
export PATH="/home/ruh/.local/bin:/home/ruh/.nvm/versions/node/v22.21.1/bin:$PATH"
bash tools/prepare.sh
npm test
npm run test:cpu
npm run test:gpu
```

`tools/prepare.sh` clones both exact upstream commits into `.cache/upstream/`, creates `.venv/`, installs the exact versions in `tools/requirements-export.txt`, verifies the checkpoint with MD5 **and** SHA-256, exports all precisions, reloads every bundle into the unmodified pinned PyTorch modules, and generates the oracles. No upstream source is patched; `model.py` is imported directly to avoid unrelated plotting/data package imports. PyTorch runs deterministic CPU evaluation with four threads.

`tools/export_evodiff.py` emits the **`evodiff-safetensors-v1`** manifest. `tensors[name]` gives dtype, original shape, and shard; INT8 records identify a colocated `<name>.__scale` companion. Matrices are F32/F16/I8, while every bias, LayerNorm parameter, and small embedding stays F32. INT8 uses `scale=max(abs(channel))/127`, nearest-even rounding, and clamping to `[-127,127]`. Quantized reference inference dequantizes weights to FP32 and accumulates in FP32. Whole tensors and their scales are packed atomically, measuring the exact serialized safetensors header. **Every shard is at most 25,000,000 bytes**, with exact `byteLength` and `sha256` recorded.

Large artifacts stay gitignored: `weights/`, `.venv/`, and `.cache/` (including checkpoint, cloned sources, and binary oracles). Only small oracle metadata and metric JSON are in `fixtures/`. Nothing is uploaded to Hugging Face.

## Measured validation

The thresholds apply to **complete final logits against the matching-precision upstream oracle**. Quantization error relative to original FP32 is reported separately. `relRMS = sqrt(sum((actual-reference)^2)/sum(reference^2))` over every element; all values must be finite. The gate is `<1e-4`, with zero skipped tests.

<!-- MEASUREMENTS_START -->
### Export sizes and quantization error versus original FP32

| Bundle | Safetensors bytes | Shards | Largest shard | Worst logit relRMS | Worst probability relRMS |
|---|---:|---:|---:|---:|---:|
| FP32 | 151,582,116 | 7 | 24,310,060 | 0 | 0 |
| FP16 | 76,004,036 | 4 | 23,788,932 | 0.00018966595 | 0.00048209911 |
| INT8 | 38,356,136 | 2 | 23,971,624 | 0.0070606557 | 0.014887602 |

### Mode Matrix: worst final-output relRMS

| Backend / precision | A: final logits | B: all 24 step logits | C: all 8 step logits | B/C sequence replay |
|---|---:|---:|---:|---|
| CPU FP32 | 8.7238858e-07 | 6.6481936e-07 | 4.6895677e-07 | Exact / exact |
| CPU FP16 | 9.2027995e-07 | 6.8693707e-07 | 4.6408574e-07 | Exact / exact |
| CPU INT8 | 9.1611263e-07 | 6.7886133e-07 | 4.4925232e-07 | Exact / exact |
| WebGPU FP32 | 9.7970306e-07 | 3.9026167e-07 | 5.7291651e-07 | Exact / exact |
| WebGPU FP16 | 9.9688812e-07 | 5.4099773e-07 | 6.5988498e-07 | Exact / exact |
| WebGPU INT8 | 1.0424989e-06 | 5.0489617e-07 | 5.7274904e-07 | Exact / exact |

Worst WebGPU error across **all intermediate trace tensors**: `2.7409107e-06`. Native `useF16=true` final-logit relRMS: `3.074963e-07`. Zero skips.

All precisions replay the same complete sequences: Mode B `MELTDCTSLQELFSSCKDPGTSVH`; Mode C `MLTAACDETSSG` (fixed motif `ACDE`).

### NVIDIA GB10 latency

| Precision | Cold engine load (ms) | L=32 warm (ms) | L=128 warm (ms) | L=256 warm (ms) |
|---|---:|---:|---:|---:|
| FP32 | 727.900 | 10.700 | 10.300 | 12.200 |
| FP16 | 384.000 | 11.400 | 11.900 | 13.900 |
| INT8 | 283.700 | 17.100 | 18.900 | 19.300 |

Warm latency is the median of three complete forwards after one warmup, including submission, completion, and logit readback. Cold engine load includes local HTTP weights, SHA verification, upload, and pipeline creation with persistent weight caching disabled; OS/driver caches may be warm.

Host: `NVIDIA GB10, 580.159.03, GPU-cb4201ed-0d40-feb9-d1ff-5766b26ed9a1`. Browser: `Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/153.0.0.0 Safari/537.36`.

Raw evidence: [export_metrics.json](fixtures/export_metrics.json), [cpu_metrics.json](fixtures/cpu_metrics.json), [gpu_metrics.json](fixtures/gpu_metrics.json).

Final command runs ([test_runs.json](fixtures/test_runs.json)):

| Command | Exit code | Elapsed seconds |
|---|---:|---:|
| `npm test` | 0 | 135.964366 |
| `npm run test:cpu` | 0 | 97.720156 |
| `npm run test:gpu` | 0 | 36.110794 |
<!-- MEASUREMENTS_END -->

Mode A includes a partially masked GFP fragment, partially masked ubiquitin, all 31 special/standard tokens, explicit padding-mask behavior, and a 257-position input that exercises every tap at the maximum dilation (including offsets ±256). It compares **all 20 trace tensors** per input, including all 16 blocks and final logits. Modes B/C compare **every step's complete `[L,31]` logits**, each sampled token, and the exact final sequence. Across three precisions this covers 300 Mode A traces and 96 sampling-step logit matrices.

The real module Worker checks transferable Float32Array results, real Cache API and OPFS offline loads with `fetch` disabled, and parity after both loads. Safari Private Mode behavior is tested by injecting storage SecurityErrors; this is a compatibility-path simulation, not a Safari browser measurement. QuotaExceededError write fallback is also tested. The browser demo is exercised in an iframe through generation, motif preservation, precision change, FASTA Blob download, progress, cancellation, and release.

GPU hygiene tests include normal completion, invalid inputs, an injected `queue.submit` exception **after the readback buffer is allocated**, an already-aborted signal, in-flight forward cancellation, and cancellation during sampling. Transient counts are zero afterward; destruction releases all tracked bytes. An externally supplied device remains usable across engines.

### WGSL portability

Seven shader variants also compile through **wgpu-native/Naga** on the GB10, including native F16. No dynamically indexed `const` arrays are used. Reproduce with:

```sh
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv/bin/python -r tools/requirements-validation.txt
.venv/bin/python tools/validate_wgsl.py
```

See `fixtures/wgsl_validation.json`. This checks the shader frontend used by the wgpu ecosystem; browser end-to-end measurements are from Chromium. [wgpu-py](https://pypi.org/project/wgpu/0.29.0/) is a development-only validation dependency.

### Hardware browser invocation

The zero-dependency Node harness starts Chromium on localhost with `DISPLAY` and `WAYLAND_DISPLAY` unset and the NVIDIA Vulkan ICD selected. It rejects software adapters and asserts NVIDIA hardware. The host GPU/driver comes from `nvidia-smi`; wgpu-native independently reports NVIDIA GB10.

```text
CHROME_PATH=/home/ruh/.cache/ms-playwright/chromium-1243/chrome-linux-arm64/chrome
--headless=new --no-sandbox --disable-gpu-sandbox --ozone-platform=headless
--enable-unsafe-webgpu --ignore-gpu-blocklist --enable-features=Vulkan
--use-angle=vulkan --use-vulkan=native --disable-vulkan-surface
--enable-dawn-features=vulkan_enable_f16_on_nvidia
```

The harness uses a short relative `.t` temporary directory within this package because Unix-domain socket paths exceed Chromium's limit with the full workspace prefix. Browser profiles, shader caches, and logs remain within this package.

## Vendor sync and denovo.run mapping

```sh
node sync-upstream.mjs /path/to/vendor/evodiff          # preview
node sync-upstream.mjs .cache/vendor-smoke --write     # tested copy
```

Only `shared/`, `webgpu/`, `cpu/`, `LICENSE`, and this README are copied. Tests import the vendored modules and verify byte equality and the absence of runtime package/Node imports. Weight manifests remain separately hostable.

| Port operation | Proposed denovo.run representation |
|---|---|
| Unconditional generation | A `SequenceSet` entry per generated sequence; metadata: OA_DM_38M, pinned revisions, precision, seed, order, uniforms, elapsed time |
| Motif / inpainting | Input `SequenceSet` plus a boolean `ResidueTrack` for fixed positions; the complementary positions map to `maskPositions` |
| Per-position logits | `ResidueTrack` with 31 named channels in tokenizer order; optional probability/entropy views derived from the logits |
| Diffusion progress | Current sequence and a boolean unmasked `ResidueTrack`, indexed by zero-based residue position |
| Export | FASTA for generated `SequenceSet`; preserve replay metadata separately for reproducibility |

This mapping is an integration proposal. The package is self-contained and does not modify denovo.run or sibling model ports.
