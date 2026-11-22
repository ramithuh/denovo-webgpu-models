# La-Proteina LD1 + AE1 for WebGPU

Standalone, dependency-free ES modules for de novo protein generation: the full
LD1 velocity network, self-conditioned sampling, and the full AE1 decoder through
atom37 coordinates and sequence logits. All neural operations and numerical
sampling updates execute WGSL in the GPU engine. A separate JavaScript CPU
backend implements the same explicit module hierarchy.

Upstream: [NVIDIA-BioNeMo/la-proteina](https://github.com/NVIDIA-BioNeMo/la-proteina/tree/cde5de3ead6e4d76f367da6dc5174be9913ef6ca),
commit **`cde5de3ead6e4d76f367da6dc5174be9913ef6ca`**. Code is accompanied by
[Apache-2.0](LICENSE); checkpoint terms are retained separately in
[WEIGHTS_LICENSE](WEIGHTS_LICENSE), the NVIDIA Open Model License Agreement.
[NOTICE](NOTICE) and [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES) retain attribution.

## Architecture and supported modes

| Component | Implementation |
|---|---|
| LD1 | 14 transformer blocks, width 768, 12 heads, head width 64, pair width 256, conditioning width 256, latent width 8 |
| AE1 decoder | 12 transformer blocks, width 768, 12 heads, pair width 256, zero conditioning width 128 |
| Attention | Pair bias, query/key LayerNorm over all 768 channels, sigmoid gating, adaptive input normalization and output scales |
| Transition | SwiGLU, expansion 4; conditioning transitions use expansion 2 |
| Output | LD1 Cα/latent velocities; AE1 20 sequence logits and 37 × 3 coordinates per residue |
| Dynamic execution | Handwritten neural modules; runtime `length` and `steps`; no ATen/FX graph capture or shape/step replay |
| Scope | One unpadded contiguous chain, integer lengths 1–500, arbitrary positive integer step count |
| Sampling | Default `sde` matches `inference_ucond_notri`; optional `ode` changes the integration mode to upstream `vf` |
| AE1 encoder | Deliberately omitted. Export and browser inference include the decoder only; generation is complete without the encoder |
| Motif / triangular models | Outside this LD1 + AE1 package |

Checkpoint configuration is authoritative. In particular, LD1 has a 46-channel
initial feature vector including `cropped_flag_seq`, which is missing from the
visible training YAML. Unconditional optional coordinate/type/cropped features
are zero. Missing first-step self-conditioning pair distances are **all zero**,
not a one-hot bin for coincident coordinates.

The default sampler uses logarithmic Cα times (`p=2`) and power latent times
(`p=2`). Noise schedules are `1/(t+0.01)` for Cα and the upstream tangent schedule
for latents. Both SDE noise scales are **0.1** inside
`sqrt(2 * g(t) * noiseScale * dt)`. Initial samples are standard Gaussian. The
pinned checkpoint's legacy `zero_com` argument is ignored by the upstream
`zero_com_noise` constructor; Cα is centered after each integration step.
Latents are not centered. Previous clean predictions `x + (1-t)*v` feed the next
step. Above `t=0.98`, upstream switches to score-scaled ODE with scale 1.5.
These details are exercised by saved exact PyTorch initial values and noise draws.

AE1 predicts positions relative to Cα, overwrites its own Cα offset with zero,
and adds the supplied Cα. Its raw coordinates include all 37 slots. The API also
returns residue-specific atom masks, masked coordinates, a sequence, and a PDB
containing the appropriate backbone and sidechain atoms.
For extreme coarse-step samples outside PDB's fixed-width coordinate range, only
the PDB representation receives a rigid translation. `pdbTranslationAngstrom`
and a `REMARK 900` record expose that offset; raw tensor coordinates are unchanged.

## API

```js
import { createLaProteinaEngine } from './webgpu/index.js';

const engine = await createLaProteinaEngine({
  baseUrl: '/models/la-proteina/weights',
  precision: 'int8',             // fp32 | fp16 | int8
  cache: 'cache',                // none | cache | opfs
  allowSoftwareFallback: true,   // production default
  // device, adapter, useF16: false, signal, onProgress,
  // read: async url => ArrayBuffer
});
try {
  const result = await engine.sample({length: 100, steps: 400, seed: 5});
  console.log(result.sequence, result.pdb, result.coords, engine.memoryUsage());
} finally {
  engine.destroy();
}
```

`createLaProteinaCPU` from `cpu/index.js` exposes the same methods. Runtime folders
`shared/`, `cpu/`, and `webgpu/` import only relative ES modules, with no npm
dependencies or Node built-ins. Node APIs are confined to development tooling.

| Method | Contract |
|---|---|
| `sample({length=100, steps=400, seed=5, mode='sde', signal, onProgress, returnTrajectory=false})` | Full LD1 sampling + AE1 decoding |
| `denoiseStep({length, ca, z, tCa, tZ, scCa?, scZ?}, options)` | Complete LD1 forward; returns Cα and latent velocities |
| `decode({length, ca, z}, options)` | Complete AE1 decoder; returns sequence, all-atom tensors, PDB |
| `memoryUsage()` | Tracked GPUBuffer `bytes`, `transientBuffers`, cumulative allocations and scratch reuse; excludes driver pipeline memory and host arrays |
| `destroy()` | Idempotent; frees every owned buffer; retains a caller-owned GPUDevice |

All numeric arrays are flattened Float32Arrays. `ca` is `[L,3]`, `z` is `[L,8]`,
`coords` and `maskedCoords` are `[L,37,3]` in **nanometers**, and `logits` are
`[L,20]` in AlphaFold residue order `ARNDCQEGHILKMFPSTWYV`. `residueTypes` and
`atomMask` are Uint8Arrays. PDB coordinates are converted to **Angstrom**.
`trajectory[k]` contains the input Cα/latent state and both velocity outputs.
`timingMs` covers sampling and decoding, excluding initial weight loading.
`denoiseStep`/`decode` options accept `signal`, `onProgress`, and
`returnIntermediates` for debugging.

For exact cross-language replay, pass `initialCa`, `initialZ`, and `draws` to
`sample`. Draws follow upstream order: Cα then latents for each stochastic step,
omitting modalities whose time exceeds 0.98. The seeded browser generator is
Mulberry32 + Box–Muller; a browser seed alone is not a PyTorch seed-equivalence
promise. `noiseScaleCa` and `noiseScaleZ` default to 0.1.

The adapter path retries high-performance acquisition and falls back to plain
`requestAdapter()`. Software adapters work by default and are reported via
`adapterInfo` and `isFallbackAdapter`. Tests explicitly set
`allowSoftwareFallback:false` and reject SwiftShader, llvmpipe/lavapipe, and WARP.
For strict caller-owned-device mode, supply its `adapter` or verified
`adapterInfo`/`isFallbackAdapter` metadata.

FP16 weights stay packed as two halves per u32, INT8 as four signed bytes per
u32 with colocated FP32 row scales. Shaders unpack only values being used.
`useF16:true` requests `shader-f16` when supported and reads FP16 matrix storage
natively; accumulation and activations remain FP32. Unsupported native F16 uses
the packed u32 path. No full FP32 weight expansion occurs in GPU memory.
GPU matrix accumulators start with the bias, matching the pinned PyTorch
`addmm` order. Exponentials use the SLEEF polynomial, and time-embedding
sine/cosine use SLEEF polynomials with split pi/2 reduction and explicit
fused rounding for the float-pair corrections.
The CPU backend stores FP32 activations and uses JavaScript double precision
for dot-product accumulation before rounding its output to FP32.

Inference on one engine is serialized; overlapping calls reject. Buffers are
checked against device size/binding limits, reused between transformer blocks,
and cleaned in `finally`. Abort is observed between layers/steps; submitted GPU
commands cannot be interrupted. A worker keeps the page responsive:

```js
import { createLaProteinaWorker } from './webgpu/worker-client.js';
const worker = await createLaProteinaWorker({baseUrl: '/weights', precision: 'int8'});
try { console.log((await worker.sample({length: 64, steps: 400})).pdb); }
finally { await worker.destroy(); }
```

## Reproduction and verification

```sh
bash tools/prepare.sh
npm test
npm run test:gpu
uv pip install --python .venv/bin/python -r tools/requirements-validation.txt
XDG_CACHE_HOME="$PWD/.cache/wgpu" .venv/bin/python tools/validate_wgsl.py
npm run serve
# Open http://127.0.0.1:8080/demo/
```

For the longer default-step audit, regenerate its separate reference manifest and
run the additional hardware suite. After all suites and Naga pass, regenerate
the measured tables and source/evidence hashes:

```sh
.venv/bin/python tools/dump_oracle.py --lengths 32 --steps 400 --precisions fp32 int8 --output long_oracles.json
npm run test:gpu:long
.venv/bin/python tools/render_metrics.py
.venv/bin/python tools/record_validation.py
```

The 400-step audit enforces the same `1e-4` threshold for every trajectory
input and velocity, final tensors, and exact sequence/atom masks. It uses the
same dynamic runtime and saved PyTorch noise as the main suite.

The exporter requires Python 3.11 and uses `uv venv` with hardlinked packages.
Only two official NGC version 1.0 checkpoint files are downloaded:

| Checkpoint | Bytes | SHA-256 |
|---|---:|---|
| `LD1_ucond_notri_512.ckpt` | 2,926,616,286 | `6d9ec013f73be6bc305876a02f2aa5dbf3183a938332bac9262aa30f8c963f66` |
| `AE1_ucond_512.ckpt` | 4,100,184,649 | `898da17022cdeaaaea7caace41c8b6fe7bfcb78be4876b0113127eb9bb1527e6` |

`tools/download_weights.py` obtains signed URLs from the NGC v2
`resources/org/nvidia/team/clara/<resource>/1.0/files` endpoint, checks the
publisher's SHA-256 against these pins, verifies the streamed download, and
renames the partial file only after validation.

`tools/export_laproteina.py` emits deterministic sharded safetensors, capped at
25,000,000 bytes including headers. It supports multipart row splits and keeps
every INT8 part beside its `.__scale` tensor. Every matrix is quantized; biases
and normalization vectors remain FP32. INT8 uses symmetric per-output-row scales
`max(abs(row))/127`, nearest-even rounding and clipping to ±127. Every exported
value is checked exactly against the intended representation and strictly loaded
back into the pinned modules. JSON manifests describe shapes, parts, byte sizes,
hashes, scope, and embedded checkpoint architecture.

`tools/dump_oracle.py` imports the original upstream neural and flow-matcher
code. Import-only adapters replace the unused training framework base class and
reject any attempted use of the unused `torch_scatter` dependency. No forward or
sampling arithmetic is replaced. Quantized references reload the actual exports
and perform FP32 arithmetic with identically dequantized weights.

The main matrix has L=32/64, T=1/3, SDE/ODE, and all three precisions (24 GPU,
8 FP32 CPU cases). An additional L=47, T=7 matrix checks an irregular shape and
step count in all precisions (6 GPU cases). Tests compare final Cα, latents,
all-atom coordinates, sequence logits, masked coordinates, and every sampled
state/velocity; residue types and atom masks must match exactly. Missing weights,
fixtures, or hardware are errors, with no skipped tests. Binary references stay
under gitignored `.cache/fixtures/`; JSON metadata records hashes and offsets.

Cache API and OPFS store verified manifests/shards. Security/quota failures fall
back to fetching, including the Safari Private Mode storage-denial path. The
tests exercise real Chromium storage plus injected permission failures; they do
not claim a Safari browser run. Real module-worker inference, normal/error/abort
cleanup, native FP16, and external-device ownership are checked. Exercised WGSL
sources are saved only under `.cache/shaders/` and validated with wgpu/Naga.

## Vendor integration and denovo.run mapping

`node sync-upstream.mjs TARGET` previews the runtime allowlist; `--write` copies
it. The intended coordinator destination is
`ProteinDesignSpace/web/static/vendor/la-proteina/`. Weights remain separately
hostable. Local verification uses `.cache/vendor-smoke`; no external vendor tree,
root package, sibling model, git commit, push, or upload is part of this port.

| Port output/action | Proposed denovo.run representation |
|---|---|
| `sequence`, `pdb`, `maskedCoords`, `atomMask` | `StructureArtifact` with sequence; convert nm coordinates to Å and record atom37 order |
| `logits` | 20-channel `ResidueTrack` in `ARNDCQEGHILKMFPSTWYV` order |
| `trajectory` | `TrajectoryFrames` for Cα; latent state and exact noise retained as replay metadata |
| Sampling request | Length, steps, SDE/ODE, seed/draws, both 0.1 noise scales, precision, checkpoint hashes and pinned commit |
| Model session | Worker lifetime; forward AbortSignal for cancellation and await `destroy()` on close |

This is a port mapping, not a modification of the denovo.run application.

<!-- measured-metrics -->

## Measured results

Hardware: **NVIDIA GB10, 580.159.03**, Chromium native Vulkan WebGPU. Values compare tensors in the original coordinate frame, without rigid alignment. All listed cases passed; **0 skipped**.

| Model | Tensors | Parameters | FP32 bytes | FP16 bytes | INT8 bytes |
|---|---:|---:|---:|---:|---:|
| LD1 | 457 | 158,402,816 | 633,666,032 | 317,233,888 | 159,920,540 |
| AE1 decoder | 384 | 127,938,304 | 511,799,128 | 256,228,352 | 129,210,140 |

Serialized bytes include safetensors headers. Strict export round-trip error is exactly zero in every precision. Largest shard: 24,959,980 bytes.

| Backend / precision | Cases | Worst final Cα relRMS | Latents | All-atom coordinates | Sequence logits |
|---|---:|---:|---:|---:|---:|
| CPU fp32 | 8 | 1.330e-05 | 5.861e-06 | 8.194e-06 | 1.705e-05 |
| GPU fp32 | 10 | 1.397e-06 | 1.099e-05 | 3.773e-06 | 1.680e-05 |
| GPU fp16 | 10 | 1.304e-06 | 9.020e-06 | 4.457e-06 | 2.181e-05 |
| GPU int8 | 10 | 1.365e-06 | 3.564e-06 | 2.405e-06 | 9.642e-06 |

The parity threshold is `relRMS < 1e-4` against the identical exported precision. Every trajectory state and velocity in these 30 GPU cases also passes; residue argmax and atom masks match exactly.

| Additional L32 / 400-step case | Worst final relRMS | Runtime (s) | Worst trajectory relRMS |
|---|---:|---:|---:|
| int8_L32_T400_sde | 2.353e-07 | 49.36 | 1.478e-05 |
| int8_L32_T400_ode | 4.003e-07 | 27.91 | 1.273e-05 |
| fp32_L32_T400_sde | 2.880e-07 | 48.69 | 1.464e-05 |
| fp32_L32_T400_ode | 5.129e-07 | 25.72 | 1.295e-05 |

All four long cases pass the same `1e-4` threshold for every trajectory input and velocity as well as final tensors, with exact sequence/atom-mask agreement. Per-step measurements are retained in `fixtures/gpu_long_metrics.json`.

The time-embedding regression covers angles through 2000 radians: GPU relRMS 0.000e+00, max absolute error 0.000e+00. WGSL uses split pi/2 range reduction to avoid large-angle native trigonometric error.

Quantization changes the model; its drift from original FP32 is measured separately below.

| Quantization drift vs FP32, worst measured relRMS | Cα | Latents | All-atom coordinates | Logits |
|---|---:|---:|---:|---:|
| fp16 | 1.257e-02 | 1.573e-02 | 1.687e-02 | 7.580e-02 |
| int8 | 1.315e-01 | 1.864e-01 | 1.300e-01 | 3.115e-01 |

These drift maxima include very short SDE/ODE trajectories; they measure model quantization effects, not browser arithmetic error or biological designability.

| Precision | Packed GPU weight bytes | Ratio to FP32 | First L32/T1 SDE call (ms) | Same call warm (ms) |
|---|---:|---:|---:|---:|
| fp32 | 1,145,364,480 | 1.0000 | 491.5 | 113.0 |
| fp16 | 573,364,736 | 0.5006 | 545.7 | 119.2 |
| int8 | 288,991,768 | 0.2523 | 564.9 | 112.0 |

Cold here means the first invocation on a new engine; the browser/driver shader cache may already be warm. Weight fetch/loading is excluded from these forward timings and recorded separately in `fixtures/gpu_metrics.json`.

Lifecycle tests pass for normal completion, invalid inputs, an injected mid-forward error, mid-forward cancellation, recovery, external GPUDevice ownership, and idempotent destruction. Every completion path leaves **0 transient buffers**, and destruction leaves **0 tracked bytes**. Real worker parity, native F16, real Cache API/OPFS cold and warm loads, and injected storage-denial fallback pass.

**462 shader variants** compiled through wgpu-native/Naga on NVIDIA GB10; 0 failures and 0 skips. Runtime WGSL uses mutable local/workgroup arrays rather than dynamically indexed const arrays.

Evidence: `fixtures/export_metrics.json`, `fixtures/cpu_metrics.json`, `fixtures/gpu_metrics.json`, `fixtures/gpu_long_metrics.json`, `fixtures/wgsl_validation.json`, and the hash-addressed binary fixture metadata.
