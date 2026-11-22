# ESM3 Open in WebGPU

Standalone browser ES modules for the complete `esm3-sm-open-v1` multimodal trunk, structure token encoder and decoder. No runtime dependencies, Node imports, WASM, ONNX runtime, or inference service. All learned operations run in WGSL; JavaScript handles inputs, loading, sampling decisions, dispatch and readback.

Source: [Biohub/evolutionaryscale ESM](https://github.com/Biohub/esm/tree/d49ee4606ea4a95a8bc7388171a6e21fefdf1c26), pinned commit `d49ee4606ea4a95a8bc7388171a6e21fefdf1c26`. Checkpoint snapshot: `EvolutionaryScale/esm3-sm-open-v1@2feda652c51c57f9797fc1b01f8d6181b38fb488`. Checkpoint digests and complete export inventory are in [fixtures/audit.json](fixtures/audit.json). Runtime transformer kernels and the sharded loader extend the verified Dyna1 port.

## Architecture and modes

| Component / mode | Implementation |
| --- | --- |
| Trunk | All 48 blocks, 1536 channels, 24 × 64 attention heads, full-channel Q/K LayerNorm, split-half RoPE, 4096-wide SwiGLU, residual scaling `sqrt(48/36)` |
| Geometric attention | Block 0, 256 vector heads, frames from N/CA/C, rotation and distance terms, chain masks, zero update for frameless queries |
| Input tracks | Sequence, structure tokens, SS8, SASA, 8 function token channels, 16 residue annotation tokens, average and per-residue pLDDT RBF projections |
| Output heads | Sequence (64), structure (4096), SS8 (11), SASA (19), function (8 × 260), residue annotations (1478), final pre-normalization embeddings (1536) |
| Structure encoder | KNN up to 16 neighbors; relative position embeddings; 2 geometric 1024-channel blocks; 128-channel projection; 4096-entry VQ codebook |
| Structure decoder | All 30 blocks, 1280 channels, 20 heads, 3584-wide SwiGLU; 6D rotation projection; N/CA/C coordinates; pLDDT, pairwise confidence logits, PAE, pTM and rigid frames |
| Sequence-only / masked fill | Forward and iterative generation, canonical output vocabulary, fixed context residues preserved |
| Inverse folding | Coordinates → VQ encoder → structure tokens plus block-0 geometric conditioning → iterative sequence fill |
| Folding | Sequence → iterative structure token fill → complete VQ decoder → backbone coordinates |
| Multitrack | Sequence + SS8 + SASA + function and residue annotation IDs; raw token tracks may be combined |
| Missing coordinates | Nonfinite atoms invalidate the whole residue frame; black-hole frame from mean valid backbone; sequence-distance KNN fallback |
| Multiple chains / packed sequences | Explicit `chain_id` for geometry and `sequence_id` for ordinary attention; `:` / `|` sequence chain breaks map to token 31 |
| PDB input | First model, selected or first chain, N/CA/C atoms, insertion codes retained, missing atoms become NaN; alternate conformer blank/A |
| Batch shape | One protein per API call; the encoder internally batches local neighborhoods |
| Function text decoder | The separate 650M-parameter function-to-InterPro/keyword decoder is audited but not exported; it is not invoked by the trunk, structure encoder, or structure decoder. Numeric function/residue conditioning and both trunk output heads are implemented. See upstream `esm3.py:get_function_decoder` and `models/function_decoder.py`. |

Coordinates include BOS/EOS rows. Slice `coordinates.subarray(9, coordinates.length - 9)` for physical residues. `structure_logits` predicts codebook tokens 0–4095; structure BOS/EOS are 4098/4097. Output logits retain special-token rows and full vocabularies. Coordinates are model predictions, without relaxation or alignment to an experimental structure.

## API

```js
import {createESM3Engine} from './webgpu/index.js';
const engine = await createESM3Engine({
  baseUrl: '/models/esm3/weights', // root containing precision directories
  precision: 'fp16',             // fp32, fp16, int8
  cache: 'cache',                // cache, opfs, none
  onProgress: ({phase,loaded,total}) => { if (phase === 'weights') console.log(loaded / total); },
});
try {
  const out = await engine.forward('NLYIQWLKDGGPSSGRPPPS');
  console.log(out.sequence_logits, out.structure_logits, out.embeddings);
  const filled = await engine.generate({sequence:'NL_IQWL_DGGPSSGRPPPS'},
    {track:'sequence', steps:8, temperature:0});
  const folded = await engine.generate({sequence:'NLYIQWLKDGGPSSGRPPPS'},
    {track:'structure', steps:8, temperature:0});
  console.log(folded.coordinates);
} finally { engine.destroy(); }
```

Inverse folding: supply `{sequence:'_'.repeat(L), coordinates}` to `generate(..., {track:'sequence'})`, with flattened `L × 3 × 3` N/CA/C coordinates. The raw interface accepts `sequence_tokens`, `structure_tokens`, `structure_coords`, `ss8_tokens`, `sasa_tokens`, `function_tokens`, `residue_annotation_tokens`, `average_plddt`, `per_res_plddt`, `chain_id`, `sequence_id`, `attention_mask`, and `residue_index`. Raw tracks include special tokens and must have matching lengths. A string sequence adds BOS/EOS. `ss8` strings and `sasa` arrays are convenience inputs padded with their upstream pad tokens.

`encodeStructure(input)` exposes VQ tokens and frames. `decodeStructure(tokens)` requires structure BOS/EOS. `forward(input, {returnIntermediates:true})` returns every block trace. `generate` uses a documented confidence-ranked cosine mask schedule, deterministic at temperature zero; positive temperatures use seeded Mulberry32 categorical draws. It preserves supplied tokens. This is a compact local generation API, not an implementation of every Forge SDK strategy.

Use `signal` on load and inference for cancellation. `memoryUsage()` reports live buffer bytes and transient counts; `destroy()` is idempotent. An external `device` stays owned by the caller. Overlapping calls on one engine reject. Backend exceptions and aborts clean up transient buffers. Device loss rejects subsequent work. WebGPU limits are checked before buffer allocation; large dispatches use a two-dimensional grid.

`webgpu/worker.js` accepts `{id, method, input, options}` messages for `load`, `forward`, `generate`, `encodeStructure`, `decodeStructure`, `cancel`, and `destroy`; it returns `{id,result}`, `{id,progress}`, or `{id,error}`. Cache API / OPFS use `esm3-v1`, verify SHA-256 on every load and support offline hits. Persistent-storage denial falls back to network loading; Cache API operations also fall back after a five-second storage timeout. Shard fetches bypass the browser HTTP cache; explicit Cache API/OPFS options manage persistence. Weights stay in browser storage until the origin cache is cleared.

The CPU reference uses the same complete model graph with pure JavaScript numerical primitives:

```js
import {createESM3CPU} from './cpu/index.js';
const cpu = await createESM3CPU({baseUrl:'/weights', precision:'fp32'});
const result = await cpu.forward('ACDEFGHIKLMN');
cpu.destroy();
```

## Precision and exported weights

| Storage | Shards | Bytes | Largest shard |
| --- | ---: | ---: | ---: |
| FP32 | 505 | 8,205,998,164 | 24,934,600 |
| FP16 | 216 | 4,135,516,508 | 24,932,552 |
| INT8 | 109 | 2,105,212,488 | 24,932,552 |

There are 892 logical tensors and 2,051,457,549 state elements across the three models, including the codebook's training buffers. The trunk has 1,401,735,748 state elements. Available disk was 96 GB before setup, with approximately 84 GiB RAM available; full FP32 bundles fit, so INT4 was unnecessary.

`esm3-safetensors-v1` manifests record source revisions, tensor shapes and dtypes, SHA-256 and byte lengths, quantization metadata, and multipart row offsets. Every shard is strictly below 25,000,000 bytes including headers. INT8 uses symmetric per-output-row scales, colocated with each tensor part. Vectors, embeddings and codebooks remain F32. Runtime weights stream shard by shard into GPU buffers; the whole complete model is then resident during inference.

All three cached checkpoints are originally **BF16**. FP32 expands those values without information loss. FP16 storage preserves nearly all these already-rounded values; its remaining rounding differences and output drift are measured. Every runtime mode uses **F32 activations and accumulation**, including optional native `shader-f16` storage loads (`useF16:true`; the default uses portable packed-half loads). Matching PyTorch references reload the exported F16/I8 weights, dequantize to F32, and run F32 arithmetic with TF32 disabled. This is storage-precision parity; it does not claim equivalence to PyTorch autocast with FP16 activations.

[fixtures/quantization.json](fixtures/quantization.json) reports drift of each precision versus original checkpoint values executed in FP32, separately from arithmetic parity. Decoder end-to-end drift includes changes in predicted discrete structure tokens; compare it as a pipeline metric, not as rigidly aligned structural RMSD. Global weight relRMS includes large EMA codebook buffers and is not a quality metric.

## Reproduce

Run from this package directory, with Node 22 and Python 3.11 available:

```sh
uv venv .venv --python 3.11
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv/bin/python -r tools/requirements-export.txt
# Source and checkpoints are pinned in tools/upstream.py.
git clone https://github.com/Biohub/esm.git .cache/upstream
git -C .cache/upstream checkout d49ee4606ea4a95a8bc7388171a6e21fefdf1c26
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/export_esm3.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/dump_oracle.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/generation_oracle.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/audit.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/validate_roundtrip.py
npm test
npm run test:cpu
npm run test:gpu
```

`tools/upstream.py` uses the explicitly cached checkpoint directory from this task. Adjust its `CACHE` path for another machine. No download, Hugging Face push, or Git commit is performed by the exporter. Large artifacts are ignored under `weights/`, `.cache/`, and `.venv/`; reference binaries live in `weights/`, and small provenance/offset JSON lives in `fixtures/`. Missing hardware, weights, source fixtures, or browser fails verification rather than skipping.

Hardware Chromium flags on the NVIDIA GB10 host:

```text
--headless=new --no-sandbox --disable-gpu-sandbox --ozone-platform=headless
--enable-unsafe-webgpu --ignore-gpu-blocklist --enable-features=Vulkan
--use-angle=vulkan --use-vulkan=native --disable-vulkan-surface
```

The harness unsets DISPLAY / WAYLAND_DISPLAY, uses the NVIDIA Vulkan ICD, rejects software adapters, and isolates Chromium profiles and precision runs. `CHROME_PATH` can select the browser. WGSL contains no dynamically indexed constant arrays; dynamic local arrays use function or workgroup storage.

## Demo and vendoring

Serve this package over localhost, for example `python -m http.server 8088 --bind 127.0.0.1`, then open `/demo/`. It supports precision selection, PDB conditioning, forward/fill/fold, progress, cancellation, result JSON download, backbone projection, and GPU release. It performs inference in a module Worker.

`node tools/vendor-sync.mjs DESTINATION` copies only runtime modules, this README, the license and an ESM wrapper. The smoke test vendors to `.cache/vendor-smoke`; no external application is modified. A denovo.run integration can import `webgpu/index.js` or use the supplied Worker, with `baseUrl` pointing to the selected immutable weight bundle root.

## Measured validation

NVIDIA GB10, driver 580.159.03, Chromium 153.0.8010.12 (ARM64 build 1243), hardware Vulkan WebGPU. Every listed error is against the matching PyTorch storage-precision reference, with `relRMS < 1e-4` required for **every recorded tensor**, including all block traces.

| GPU storage / loader | Sequence logits relRMS | Decoded N/CA/C relRMS | Iteratively folded coordinates relRMS | Maximum trace relRMS |
| --- | ---: | ---: | ---: | ---: |
| fp32 | 2.741e-7 | 1.054e-6 | 5.966e-7 | 3.421e-6 |
| fp16 | 2.588e-7 | 1.023e-6 | 8.320e-7 | 2.332e-6 |
| fp16-native | 2.588e-7 | 1.023e-6 | 8.320e-7 | 2.332e-6 |
| int8 | 2.636e-7 | 7.591e-7 | 7.407e-7 | 2.325e-6 |

| GPU mode | Cold weight load (s) | Warm L=20 (ms) | Warm L=64 (ms) | Warm L=128 (ms) |
| --- | ---: | ---: | ---: | ---: |
| fp32 | 31.87 | 91.4 | 198.4 | 355.4 |
| fp16 | 15.82 | 77.1 | 155.5 | 276.3 |
| fp16-native | 16.02 | 78.8 | 163.5 | 261.9 |
| int8 | 7.77 | 66.8 | 143.9 | 267.5 |

Lengths exclude BOS/EOS. Warm values are medians of three full forwards after a separate shape-specific warmup; include command preparation/readback, exclude weight loading. Cold weight loads use local HTTP without a browser weight-cache hit. The GPU was not exclusively reserved.

CPU maximum trace relRMS is **4.227e-6**. All three PyTorch strict bundle reloads reproduce their reference final outputs bit for bit. The sequence, masked sequence, coordinate-conditioned inverse folding, multitrack, decoder and three iterative generation trajectories all pass. All modes check final logits/embeddings/coordinates and intermediate blocks; generation checks every step’s logits, chosen tokens, fill order and final decoded coordinates.

| Storage drift versus original BF16 checkpoint expanded to FP32 | Sequence logits relRMS | End-to-end decoded coordinates relRMS |
| --- | ---: | ---: |
| fp16 | 2.047e-7 | 5.961e-7 |
| int8 | 1.785e-3 | 4.047e-2 |

INT8 drift is separate from arithmetic parity; coordinates can also change when predicted structure token IDs change.

| Required command | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| `npm test` | 7 | 0 | 0 |
| `npm run test:cpu` | 1 | 0 | 0 |
| `npm run test:gpu` | 1 | 0 | 0 |

Lifecycle checks cover successful inference, injected mid-run errors, mid-run aborts, failed shard loads, idempotent destruction, and continued external-device use. Cache API/OPFS offline reads, private-storage denial fallback, module-Worker folding, the demo and vendor-sync smoke test pass. The demo uses portable packed-half loads by default; native FP16 is separately tested in the main-thread engine.

Detailed evidence: [validation.json](fixtures/validation.json), [gpu.json](fixtures/gpu.json), [cpu.json](fixtures/cpu.json), [browser.json](fixtures/browser.json), [roundtrip.json](fixtures/roundtrip.json). Runtime source digests and exact suite summaries are in the completion record. No Hugging Face upload, commit, push, or external application deployment was performed.
