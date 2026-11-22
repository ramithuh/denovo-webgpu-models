# FrameFlow in browser WebGPU

A standalone ES module implementation of the complete FrameFlow PDB-checkpoint
forward pass, self-conditioning, dynamic SE(3) flow integration, and backbone PDB
export. Runtime modules have **zero npm dependencies and zero Node imports**.
There is no ONNX runtime, Python inference service, captured ATen graph, or
length/step-specific execution plan.

Upstream: [microsoft/protein-frame-flow](https://github.com/microsoft/protein-frame-flow/tree/f50d8dbbdae827be291e9f73d732b61b195f8816),
commit **`f50d8dbbdae827be291e9f73d732b61b195f8816`**, MIT.
OpenFold-derived IPA, rigid-transform algorithms and geometry retain Apache-2.0
attribution in `NOTICE` and `LICENSE-APACHE`.

**Accuracy scope:** the recorded parity matrix covers L=32/64, T=2/20,
FP32/FP16/INT8, forward calls with/without self-conditioning, and fixed-frame
motif sampling at T=3. Some additional schedules exceed the requested `1e-4`
threshold because floating-point differences cross hard distance-bin boundaries.
These failures are disclosed below; this package does not promise a uniform
`1e-4` bound over every seed and integration schedule.

## Modes

| Mode | Status with the selected PDB checkpoint |
|---|---|
| Unconditional backbone generation | Complete network and sampling implementation; dynamic L and integer T≥2 |
| Self-conditioning | Supported, on by default; optional `trans_sc` for individual forward calls |
| SO(3) annealing | Exponential-rate Euler (default rate 10) and linear schedule |
| Fixed-frame motif scaffolding | Pinned fixed-mask sampling path implemented and parity tested |
| Learned amortized motif model | Requires upstream's separate `pdb_amortization` checkpoint; not included |
| Gradient-guided motif scaffolding | Broken in pinned upstream: missing `so3_utils.sample_wrapped_normal`; explicitly rejected |
| Multichain, amino-acid design, sidechain prediction | Not provided by this backbone package |

See [ARCHITECTURE.md](./ARCHITECTURE.md) for module dimensions, pinned-source
behavior, exact mode distinctions, and the reproduced guidance failure. Fixed
frames preserve a motif; the unconditional PDB weights are not a trained
amortization model. No PDB datasets, ProteinMPNN, folding, or evaluation pipelines
are downloaded.

## Reproduce

Run from this directory. Python is used only for export and reference tensors.

```sh
bash tools/prepare.sh
npm run test:gpu
npm run test:cpu
npm test

# Independent Firefox wgpu/Naga-compatible shader validation on hardware:
uv pip install --python .venv/bin/python -r tools/requirements-validation.txt
node tools/dump_wgsl.mjs
.venv/bin/python tools/validate_wgsl.py
```

`prepare.sh` creates `.venv` with `uv`, checks out the exact upstream commit in
`.cache/upstream`, downloads the official Zenodo weights archive, extracts only
`weights/pdb/published.ckpt`, exports all three precisions, and records oracle
fixtures. It does not commit, publish, upload, or modify a sibling model.

Official archive:
[Zenodo record 12776473, weights.tar.gz](https://zenodo.org/records/12776473/files/weights.tar.gz?download=1).

| Artifact | Verified digest |
|---|---|
| Archive SHA-256 | `c36f992dbc36682049a9aa06cf204146125fcff22a3e5316339982330942b12e` |
| Archive published MD5 | `f08937bbf1bbf4889771e6eee2451c53` |
| Selected checkpoint SHA-256 | `7d0906e0184e87d1101a2be4c24c530d434e9fac01c1d777d65a63fccd1c958a` |

Upstream Python, checkpoint/export binaries, virtualenvs, browser profiles,
shader caches, and binary little-endian FP32 oracles remain gitignored. Small
JSON manifests and measured results are in `fixtures/`. Reference generation
executes the pinned `FlowModel` and `Interpolant.sample` directly. The export
helper fails if a training-only `torch_scatter` routine is used; no inference
arithmetic is substituted by that import shim.

Tests **fail**, rather than skip, if weights, fixtures, Chromium, or hardware
WebGPU are absent. Set `CHROME_PATH` to a hardware-WebGPU Chromium executable;
the default matches this machine's cached ARM64 Chromium. The harness uses
native Vulkan, ephemeral loopback ports, and profiles/temp directories under
`.cache/`. SwiftShader, llvmpipe, lavapipe, WARP, software, and fallback adapters
are explicitly rejected in GPU parity tests. Naga validation was run separately
through wgpu-native on NVIDIA GB10/Vulkan; a Firefox browser was not exercised.

## Weights and memory

382 named tensors, **16,779,012 parameters**. Matrices are FP32, packed FP16, or
symmetric per-row INT8; vector weights and biases stay FP32. INT8 scales use FP32
and are colocated with the corresponding tensor part. Shards are bounded by
**25,000,000 bytes including the safetensors header**. The exporter supports row
splits and exact reassembly when a tensor exceeds a shard. This checkpoint does
not require splitting an individual tensor.

| Precision | Safetensors bytes | Resident weight bytes | Relative weight VRAM | Reload |
|---|---:|---:|---:|---|
| FP32 | 67,155,776 | 67,116,048 | 1.0000 | Exact |
| FP16 | 33,746,176 | 33,707,536 | 0.5022 | Exact quantized tensors |
| INT8 | 17,276,704 | 17,222,240 | 0.2566 | Exact dequantized tensors |

“Exact reload” means bitwise equality to the exporter’s expected quantized
PyTorch state, checked with strict model-state loading. Quantization drift versus
FP32 is a different measurement and appears per output in `fixtures/matrix.json`.
The GPU reads FP16/INT8 directly from packed buffers while multiplying;
**loading never allocates full-size FP32 copies of those matrices in VRAM**.

`memoryUsage()` separates `weightBytes`, pooled `scratchBytes`, active
`transientBuffers`, total `bytes`, and `peakBytes`. The recorded L32/L64 suite
retained 58,677,080 scratch bytes for reuse, independent of weight precision.
The precision ratios apply to resident weights, not total weights-plus-scratch.
Normal, injected-error, and aborted calls leave zero active transient buffers.
Destroying twice is safe and leaves **zero bytes**. Owned devices are destroyed;
external devices remain usable. Every allocation checks buffer and storage
binding limits. Calls on one engine are serialized by the caller; overlapping
forward/integration calls are rejected.

## Browser API

```js
import { createFrameFlowEngine } from './webgpu/index.js';

const engine = await createFrameFlowEngine({
  baseUrl: '/models/frameflow/weights', // contains fp32/, fp16/, int8/
  precision: 'fp16',
  cache: 'cache',                      // 'none' | 'cache' | 'opfs'
  allowSoftwareFallback: true,         // production default
  onProgress: event => console.log(event),
});
try {
  const controller = new AbortController();
  const result = await engine.sample({
    length: 64,
    steps: 100,
    selfCondition: true,
    schedule: 'exp',
    expRate: 10,
    signal: controller.signal,
    onProgress: event => console.log(event),
    keepTrajectory: false,
  });
  // Flat Float32Array outputs, translations/coordinates in Å:
  console.log(result.pred_trans);     // [L,3]
  console.log(result.pred_rotmats);   // [L,3,3], row-major
  console.log(result.coords);         // [L,5,3], N,CA,C,O,CB
  console.log(result.pdb);
  console.log(engine.adapterInfo, engine.isFallbackAdapter);
  console.log(engine.memoryUsage());
} finally {
  engine.destroy();
}
```

The factory also accepts an external `device`, `useF16`, `signal`, and injected
`read(url) -> Promise<ArrayBuffer>`. `useF16: true` uses native f16 storage reads
when supported; otherwise FP16 is decoded from packed u32 pairs. Accumulation
remains FP32 in both cases. The default path uses no optional GPU feature.

Production retries high-performance adapter acquisition and then tries plain
`requestAdapter()`. It permits software adapters by default and exposes the
metadata. Setting `allowSoftwareFallback: false` requires verifiable hardware,
including for externally supplied devices.

`forward(input, { signal, onProgress })` accepts:

| Field | Shape / meaning |
|---|---|
| `length` | L; inferred from `trans_t` if omitted |
| `trans_t`, `rotmats_t` | Flat FP32 arrays `[L,3]` and `[L,3,3]` |
| `so3_t`, `r3_t` | Numbers in [0,1]; default `t` or 0.01 |
| `trans_sc` | Optional self-conditioning translations `[L,3]` |
| `res_mask`, `diffuse_mask` | Binary `[L]`; default all ones |
| `res_idx` | Float32 residue indices `[L]`; default 0…L−1 |

Forward returns `pred_trans`, `pred_rotmats`, and the idealized zero-psi `coords`.
Sampling applies upstream's neighbor-based oxygen correction before exporting
all five backbone atoms. The checkpoint predicts frames only; PDB alanine labels
do not represent a designed sequence.

Supply `sample({ initial, steps, ... })` to replay exact initial frames. A seeded
`random` callback can generate repeatable initial noise on the main-thread API;
for Worker replay, supply saved initial frames. The initial SO(3) sampler is
uniform and R3 noise is centered with scale 10 Å. PyTorch and JS random streams
are not asserted identical. The pinned final-call time reuse and FP32 time-grid
rounding are deliberately preserved.

For a motif, provide `initial.diffuse_mask` with zeros at fixed positions, or
`motif: { mask, trans, rotmats }` with ones selecting motif positions and full-length
frame arrays. `res_idx` can preserve residue numbering gaps. Scaffold placement
and PDB-to-frame/contig interpretation belong to the caller. Requesting
`guidance: true` produces an explicit unsupported-pinned-upstream error.

### Worker and CPU

```js
import { createFrameFlowWorker } from './webgpu/worker.js';
const worker = await createFrameFlowWorker({
  baseUrl: '/models/frameflow/weights', precision: 'int8', cache: 'opfs',
});
try {
  const result = await worker.sample({ length: 64, steps: 100 });
} finally {
  await worker.destroy();
}
```

The Worker exposes async `forward`, `sample`, `memoryUsage`, and `destroy`,
progress messages, and AbortSignal cancellation. CPU reference:
`import { createFrameFlowCPU } from './cpu/index.js'`; factory arguments mirror
weight-loading options. Its arithmetic uses Float32Array activations with JS
number accumulators. It executes the complete model and is intended for
verification, not interactive large-protein generation.

Cache API and OPFS persist only downloaded bytes. Both were tested cold and warm;
warm reads required zero network requests. Denied storage was simulated with
`SecurityError` exceptions to exercise Safari Private Mode-style fallback.
Failure to open/write persistent storage falls back to normal fetches; Web Crypto
SHA-256 validation remains mandatory. This is not a claim of testing Safari itself.

## Recorded parity and timing

The matrix contains 36 distinct GPU cases plus three warm repeats: twelve per
precision across two lengths, unconditional forward/self-conditioning variants,
unconditional T=2/20 trajectories, fixed-frame motif forward calls, and motif
T=3 trajectories. Assertions cover **every saved clean-step frame**, final
translations, final rotation matrices, and all five reconstructed atoms.
Tolerance is `sqrt(sum((actual-reference)^2)/sum(reference^2)) < 1e-4`;
non-finite values fail. PyTorch FP16/INT8 references dequantize the identically
exported matrices and execute in FP32.

<!-- METRICS_START -->
| Precision | GPU maximum relRMS (all checked tensors/steps) | Cold L32 forward ms | Warm L32 forward ms | L64/T20 sample ms |
|---|---:|---:|---:|---:|
| FP32 | 2.378e-05 | 280.0 | 14.6 | 570.2 |
| FP16 | 2.085e-05 | 74.1 | 14.7 | 609.4 |
| INT8 | 5.609e-05 | 57.1 | 18.1 | 630.2 |

FP32 CPU: **12 complete cases**, maximum relRMS **1.728e-05**. Hardware: NVIDIA GB10, driver 580.159.03, Chromium native Vulkan. All 22 captured shader variants compiled through Naga/wgpu-native; zero skipped validations.
<!-- METRICS_END -->

Timings include JS dispatch, GPU execution, and readback. “Cold” means the first
L32 forward on that engine; drivers may reuse compiled shader caches across
engines. Warm timing is another L32 forward after the full matrix. These are
wall times on this host, not kernel timestamp queries or cross-device guarantees.
Full details: `fixtures/export_metrics.json`, `gpu_metrics.json`,
`cpu_metrics.json`, `runtime_metrics.json`, and `wgsl_validation.json`.

### Known numerical failures beyond the verification matrix

The `1e-4` bound is **not universal**. With the same seed 123, exploratory
L64/T12 runs produced about `0.00421` FP16 and `0.01399` INT8 relative RMS in final
translations. Their single forwards with exact reference inputs remain close;
tiny trajectory differences cross hard distogram bins and accumulate. Coarse
T5 runs also exceeded the bound. These are genuine remaining accuracy limits,
not skipped tests or accepted parity successes. See `numerical_stress.json`.

To regenerate and strictly check that separate stress matrix:

```sh
.venv/bin/python tools/dump_oracle.py --steps 12 --prefix stress_
node tools/diagnose_stress.mjs   # nonzero exit if any stress result exceeds 1e-4
```

The standard verification matrix is explicit in `fixtures/matrix.json` and
`motif_matrix.json`. It does not imply that all other schedules meet the same
numerical threshold. No reference tensor is read by the production runtime.

## Demo and vendor integration

```sh
npm run demo                  # http://127.0.0.1:8095/
PORT=8096 npm run demo
node sync-upstream.mjs /path/to/ProteinDesignSpace --dry-run
node sync-upstream.mjs /path/to/ProteinDesignSpace --weights
```

The static demo uses a module Worker, precision/length/step controls, progress,
cancellation, a draggable Cα trace, and a five-atom PDB download. The browser test
also generates an independent L=17 example. Serve modules and safetensors with
HTTPS or localhost; no npm install is needed by a production browser.

Sync targets `ProteinDesignSpace/web/static/vendor/frameflow/`, copying the
standalone modules, demo, metadata and licenses. `--weights` additionally copies
only exported precision bundles. It never copies Python, virtualenvs, the
checkpoint archive, or `.cache`. No external vendor tree was modified in this task.

| denovo.run port | Proposed mapping |
|---|---|
| Generator settings | L, T, minT, SO(3) schedule/rate, precision, self-conditioning |
| `StructureArtifact` input | Optional initial/fixed backbone frames; caller converts PDB/contigs to frames |
| `ResidueTrack` input | Presence mask, diffuse/fixed mask, residue indices |
| `StructureArtifact` output | Frames, N/CA/C/O/CB coordinates, Å units, atom order, PDB |
| Trajectory output | Optional clean predicted frames per integration step |
| `Collection` | Repeated calls sharing resident weights |
| Provenance | Commit, checkpoint SHA-256, precision, integration settings, initial-frame replay identity |

This is an integration mapping, not a deployed denovo.run change. Generated
backbones still require separate sequence design and experimental evaluation.
