# ESMC-300M WebGPU

Complete browser-native ESMC-300M inference in dependency-free ES modules and WGSL. All **332,997,184 parameters in 308 tensors** are retained. Includes a JavaScript CPU correctness reference, a static demo, and deterministic upstream oracles for Trp-cage (20 aa) and ubiquitin (76 aa).

Upstream code: [Biohub/esm at `26b0bc2b771e3e419ea74f445a5f35cc094a1509`](https://github.com/Biohub/esm/tree/26b0bc2b771e3e419ea74f445a5f35cc094a1509). We pin the MIT-licensed implementation immediately before the upstream Transformers API migration. Weights: [biohub/esmc-300m-2024-12 at `7f10b20ae75017b2dbc884070e03434515709a8d`](https://huggingface.co/biohub/esmc-300m-2024-12/tree/7f10b20ae75017b2dbc884070e03434515709a8d). Original checkpoint SHA-256: `323dff9fbf3fef297a74f4f18b6528e6f2e599b0bcf72b6927516804015becea`. Both code and model are MIT-licensed by Biohub. See [LICENSE](LICENSE) and the [upstream third-party notices](https://github.com/Biohub/esm/blob/26b0bc2b771e3e419ea74f445a5f35cc094a1509/THIRD_PARTY_NOTICE.md).

Architecture: 64×960 embedding; **30** pre-LN transformer blocks; **15 heads × 64 channels**; full-channel Q/K LayerNorm; split-half rotary embeddings; SwiGLU width **2560**; residual updates divided by `sqrt(30/36)`; final bias-free LayerNorm; biased `960→960`, exact GELU, affine LayerNorm, biased `960→64` MLM head. LayerNorm epsilon is `1e-5`. GELU's erf is evaluated with an approximation whose measured layer error is covered by the same `1e-4` trace gate. No layers or output heads are pruned. Detailed input/mode exclusions and upstream source evidence are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Reproduce the bundles and oracles

All downloaded data, build caches, Python environments, and browser profiles stay inside this checkout. Python 3.12 development headers are needed to build Biotite on ARM64; uv's managed Python includes them.

```bash
mkdir -p .cache/tmp
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
export TMPDIR="$PWD/.cache/tmp"
export HF_HOME="$PWD/.cache/huggingface"
export XDG_CACHE_HOME="$PWD/.cache/xdg"
uv python install 3.12.12 --no-bin
uv venv .venv --python 3.12.12
uv pip install --python .venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match -r tools/requirements-export.txt
.venv/bin/python tools/download_weights.py
.venv/bin/python tools/export_esmc.py
.venv/bin/python tools/dump_oracle.py
.venv/bin/python tools/quality_metrics.py
.venv/bin/python tools/verify_provenance.py
```

The downloader verifies immutable source archive and checkpoint SHA-256 digests. Upstream files remain untouched. The exporter strictly reloads every bundle into the original model through the oracle script. FP32 tensors must be bit-identical to the original checkpoint before running any oracle. FP16 and INT8 are dequantized into the same CPU FP32 upstream model. [Export metrics](fixtures/export_metrics.json) therefore measure storage quantization independently of runtime arithmetic.

`esmc-safetensors-v1` manifests include model/precision, upstream commit, source revision, parameter/tensor counts, config, per-shard `byteLength`, SHA-256 and tensor mapping. Shards include headers in the **25,000,000-byte** cap. Multipart matrices use contiguous `parts` with `row_start`; synthetic split-tensor tests exercise reconstruction even though this model's largest FP32 matrix fits the cap. Embeddings and all one-dimensional parameters remain F32. Other matrices use F32, F16, or symmetric per-output-row I8 (`max(abs(row))/127`); each F32 scale companion shares its data shard. Activations and accumulation remain F32 in every GPU variant.

<!-- GENERATED_EXPORT_TABLE -->

| Bundle | Shards | Bytes including headers | Weight relRMS |
| --- | --- | --- | --- |
| FP32 | 61 | 1,332,026,360 | 0.000e+00 |
| FP16 | 30 | 666,505,040 | 1.906e-04 |
| INT8 | 15 | 334,952,848 | 8.558e-03 |

Numerical quantization error against original FP32 (FP32 round trip is exactly zero in every mode):

| Precision | Protein | Embeddings | MLM logits | WT LLR | Masked LLR | Fill logits | Naturalness | Pseudo-perplexity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP16 | trp_cage | 2.988e-04 | 1.416e-04 | 7.207e-04 | 7.613e-04 | 3.791e-05 | 1.042e-04 | 2.842e-04 |
| FP16 | ubiquitin | 4.781e-04 | 2.162e-04 | 2.086e-04 | 2.194e-04 | 2.296e-04 | 5.711e-04 | 2.269e-05 |
| INT8 | trp_cage | 1.358e-02 | 3.688e-03 | 2.757e-02 | 2.808e-02 | 4.223e-03 | 3.355e-05 | 9.151e-05 |
| INT8 | ubiquitin | 2.012e-02 | 8.403e-03 | 8.483e-03 | 8.581e-03 | 1.473e-02 | 1.383e-02 | 5.481e-04 |

<!-- END_EXPORT_TABLE -->

All four selected masked-fill top-1 predictions are unchanged in FP16 and INT8. Fill probability error and agreement are recorded in [quality_metrics.json](fixtures/quality_metrics.json).

Oracles contain inputs and sequence IDs, input embedding, every block output, final LayerNorm, MLM head stages and logits, and all task outputs. Unmasked and multi-mask runs keep full spatial traces. Each of the **96** single-mask passes keeps every block's masked-residue vector and all output stages; this makes complete sweeps reviewable without storing multi-gigabyte traces. All 30 attention maps are tested for unmasked, padded and packed-token fixtures. Binary files are deterministic little-endian F32/I32; metadata records shape, offset, length and SHA-256. Only these fixtures are committed, never weights or environment files.

## Runtime API

```js
import {createEsmcEngine} from './webgpu/index.js';
const controller = new AbortController();
const engine = await createEsmcEngine({
  baseUrl: '/weights/esmc-300m', // contains fp32/, fp16/, int8/
  precision: 'int8',
  cache: 'cache',             // 'opfs' or 'none' also supported
  useF16: false,              // core unpack2x16float; true uses shader-f16 if available
  signal: controller.signal,
  onProgress: ({phase, loaded, total}) => console.log(phase, loaded, total),
});
try {
  const result = await engine.run('NLYIQWLKDGGPSSGRPPPS', {
    mode: 'masked',           // embeddings, logits, wildtype, masked, naturalness, fill
    signal: controller.signal,
    onProgress: ({loaded, total}) => console.log(loaded, total),
  });
  console.log(result.llr, result.naturalness, result.pseudoPerplexity);
} finally {
  engine.destroy();
}
```

`forward(sequenceOrTokens, options)` returns flat typed arrays: `logits` is `(L+2)×64`, `tokenEmbeddings` is `(L+2)×960`, `embeddings` is `L×960`, `pooled` is a 960-element residue mean excluding BOS/EOS. `llr` and `canonicalProbabilities` are `L×20` in **ACDEFGHIKLMNPQRSTVWY** order, `predictions` contains canonical token IDs, and `wtLogProbs` uses the full 64-token softmax. Outputs include `timingMs`. Buffers are independent of GPU residency and can be transferred to another thread.

`returnIntermediates: true` returns all 30 pre-final-LN block states and major head stages; `hiddenLayer: 0..29` returns one selected full token hidden state. `returnAttentions: true` returns all 30 `15×T×T` attention arrays. `traceRow` bounds hidden/head traces to one token; requested attention maps retain their full shape. Token input optionally accepts `sequenceIds`; attention is allowed exactly where IDs match, as upstream. Without explicit IDs, padding is separated using `token != PAD`. `batch(sequences, options)` executes independent inputs sequentially.

| Mode | Semantics |
| --- | --- |
| `embeddings` | Per-residue and mean-pooled final LN embeddings; optional pre-LN hidden layer |
| `logits` | Full 64-token MLM logits, including BOS/EOS |
| `wildtype` | One unmasked pass; `LLR[i,a] = logit[i,a] − logit[i,WT]` |
| `masked` | One pass for every masked residue; complete L×20 masked-marginal LLR matrix |
| `naturalness` | Same complete sweep; mean masked WT log probability; pseudo-perplexity = exp(−mean log p) |
| `fill` | `positions: [0, 5, ...]` masks positions simultaneously and predicts canonical top-1 replacements; returns `filledSequence` |

Scoring and fill accept canonical wild-type sequences. The lower-level `forward` accepts uppercase noncanonical letters, gap/chain tokens and `_`/`<mask>` placeholders. `tokenize()` itself matches upstream's character tokenizer (including unknown characters) exactly; public inference rejects whitespace/lowercase/structure text rather than silently changing the biological input. The demo explicitly uppercases and removes whitespace. The display string maps unknown or interior special tokens to `X` and masks to `_`, preserving one character per output row; `tokens` preserves exact IDs. Maximum sequence length is an admission policy of **1022 internal tokens + BOS/EOS**, not a claim about maximum biological model context. Structure files, geometry, molecular context, and sidechain packing are outside ESMC's sequence-only graph.

`scoreMutations(sequence, llr, [{position, from, to}, ...])` from `shared/index.js` adds valid point-mutant LLRs, including multiple mutants. This additive score does not model epistasis. Masked-task `onTrace(position, trace)` receives each pass's intermediate tensors; `returnIntermediates: true` collects them as an array.

`createEsmcCPU({baseUrl, precision:'fp32', read})` has the same task interface and synchronous `forward()`. Its dot products use JS Number accumulation with F32 storage; it is a correctness reference and is much slower than WebGPU. Node-specific file readers belong in the caller/tests, never the runtime.

The loader validates manifest dimensions, contiguous multipart rows, parameter counts, shard lengths and SHA-256 via Web Crypto. It supports injected `read(url) → ArrayBuffer`, browser fetch, Cache API, and OPFS. Storage denial or quota errors fall back to network. Missing or corrupted data fail. Web Crypto is required. Cache entries are keyed by the bundle URL; use revisioned URLs for deployed weights.

`device` can be supplied externally; engine destruction leaves it alive. Otherwise the engine requests a hardware adapter and owns/destroys its device. Software adapters are rejected. `destroy()` is idempotent. `memoryUsage()` returns `bufferCount`, `transientBuffers`, and `totalBytes`: resident weights remain after inference, transients must be zero, and destruction must produce zero bytes. Scratch allocations are reused across blocks and destroyed in `finally`, including errors and cancellation. Individual allocations are checked against `maxBufferSize` and `maxStorageBufferBindingSize` before dispatch. In-flight GPU work can finish before cancellation returns. Reentrant tasks and forwards reject. Device loss rejects subsequent forwards.

## Verification on NVIDIA GB10 ARM64

```bash
npm test
npm run test:cpu
npm run test:gpu
```

Recorded results: **`npm test` 13/13**, **`npm run test:cpu` 1/1**, and **`npm run test:gpu` 5/5**, with zero failures, cancellations, or skips. The CPU case checks the entire mode matrix and all 96 mask passes; it took about 22 minutes. [validation_summary.json](fixtures/validation_summary.json) records command totals, transcript hashes, and runtime hashes.

Tests run with Node 22.21.1 and NVIDIA driver 580.159.03. Tests fail, never skip, when weights, fixtures, Chromium, WebGPU, native F16 support for its explicit test, or a hardware adapter are unavailable. There are no npm runtime dependencies or installation steps. Browser harnesses bind only `127.0.0.1` **port 0**. The tested Chromium executable is `~/.cache/ms-playwright/chromium-1243/chrome-linux-arm64/chrome`; `CHROME_PATH` can override it.

```text
--headless=new --no-sandbox --disable-gpu-sandbox --ozone-platform=headless
--no-first-run --no-default-browser-check --disable-background-networking
--disable-component-update --disable-sync --disable-default-apps
--enable-unsafe-webgpu --ignore-gpu-blocklist --enable-features=Vulkan
--use-angle=vulkan --use-vulkan=native --disable-vulkan-surface
--enable-dawn-features=vulkan_enable_f16_on_nvidia
```

The harness unsets DISPLAY/WAYLAND_DISPLAY and uses `/usr/share/vulkan/icd.d/nvidia_icd.json` when present. Profiles, XDG storage and temporary files use project-local `.cache/`. The adapter reports vendor `nvidia` and `isFallbackAdapter=false`; `nvidia-smi` identifies **NVIDIA GB10**. No exclusive GPU reservation was used.

<!-- GENERATED_MODE_TABLE -->

All entries passed on both proteins. Entries are the **maximum measured relRMS** across the corresponding traces/outputs.

| Mode / coverage | CPU FP32 | GPU FP32 | GPU FP16 core | GPU INT8 | GPU FP16 native |
| --- | --- | --- | --- | --- | --- |
| Embeddings, all 30 blocks + final LN | 3.077e-06 | 2.840e-06 | 2.722e-06 | 2.638e-06 | 2.722e-06 |
| MLM logits and head stages | 2.817e-06 | 2.937e-06 | 2.790e-06 | 2.619e-06 | 2.790e-06 |
| Wild-type L×20 LLR | 1.251e-05 | 6.552e-06 | 6.261e-06 | 5.762e-06 | 6.261e-06 |
| Masked L×20 LLR + every sweep block | 1.319e-05 | 7.834e-06 | 6.807e-06 | 6.940e-06 | 6.807e-06 |
| Naturalness and pseudo-perplexity | 9.090e-06 | 7.619e-06 | 4.500e-06 | 1.716e-06 | 4.500e-06 |
| Masked fill, all blocks + probabilities | 1.220e-05 | 3.890e-06 | 4.218e-06 | 4.652e-06 | 4.218e-06 |
| Attention maps + padding/packed inputs | 3.678e-06 | 6.629e-06 | 4.020e-06 | 7.132e-06 | 4.020e-06 |

<!-- END_MODE_TABLE -->

Every tested intermediate and final output must have **relRMS < 1e-4**. FP32 uses original upstream FP32 oracles. FP16/INT8 use the identically quantized upstream oracles; their separate end-to-end quantization errors remain visible above. Padding/packing, token replay, fill predictions, WT zero diagonal, and selected hidden layers are also checked. Lifecycle tests cover successful calls, injected mid-inference errors, in-flight abort, oversize admission, idempotent release and external-device survival. Real module Workers execute the complete model, reload both Cache API and OPFS offline with fetch blocked, and exercise storage SecurityError/QuotaExceededError fallback.

<!-- GENERATED_TIMING_TABLE -->

| Storage / loader | Cold load (ms) | Trp-cage warm (ms) | Ubiquitin warm (ms) |
| --- | --- | --- | --- |
| fp32 | 4907.8 | 26.6 | 47.2 |
| fp16 | 2525.1 | 24.2 | 43.4 |
| int8 | 1308.7 | 23.0 | 42.8 |
| fp16_native | 2428.1 | 23.2 | 41.5 |

<!-- END_TIMING_TABLE -->

Warm values are medians of three complete forward calls, including command encoding/readback and excluding weight load and first pipeline compilation. Masked scoring requires L forward calls. Full values, per-tensor metrics and run timings live in `fixtures/*metrics.json`.

## Demo and vendor contract

```bash
node tools/serve.mjs
# Open the printed ephemeral localhost URL.
node sync-upstream.mjs .cache/vendor-smoke        # dry-run
node sync-upstream.mjs .cache/vendor-smoke --write
```

The demo exposes precision/mode selection, progress, cancellation, matrix/embedding plots, CSV/FASTA downloads, and Release GPU. Browser tests execute every mode and validate downloads, cancellation and release. Vendor synchronization copies only `shared/`, `webgpu/`, `cpu/`, `LICENSE`, and this README. References to `fixtures/` and `ARCHITECTURE.md` refer to this source checkout; these evidence files are intentionally omitted from a vendor copy. The destination is explicit, dry-run is the default, and symlink destinations are rejected. This task smoke-tests `--write` only inside `.cache/vendor-smoke`; no external host checkout is touched.

Proposed denovo.run mapping:

| Artifact | Mapping |
| --- | --- |
| `SequenceSet` input | Sequence + stable sequence ID, optional zero-based masked positions |
| `StructureArtifact` input | Explicit upstream sequence extraction adapter; coordinates never condition ESMC |
| `ResidueTrack` output | 960-dimensional embedding, 20-channel LLR/probability track, or masked WT log p, aligned by sequence position |
| `SequenceSet` output | Filled sequence with source/mask provenance |
| `Collection` output | Per-sequence pooled embeddings, naturalness/pseudo-perplexity, precision and model revision, grouped residue tracks |

No host application files are modified by this package. Model assets remain an independently served, revisioned bundle.
