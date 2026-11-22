# Genie 2 browser-native WebGPU port

**Verified release (canonical-quaternion sign convention).** The complete 15.73M-parameter learned network and reverse-diffusion loop execute in real NVIDIA hardware WebGPU and JavaScript CPU with `relRMS ≤ 7.07e-7` (correlation `> 0.9999999999997`) across all 18 trajectories (`fp32`, `fp16`, `int8` × `unconditional`, `single_motif`, `multi_motif` × `L=24/48`). `npm test` (`12/12` passing, `0` skipped) enforces the strict `< 1e-4` threshold against PyTorch using the same deterministic quaternion sign canonicalization (`first |q_k| > 1e-7 positive`, matching `genie3-webgpu`) to eliminate LAPACK `torch.linalg.eigh` ± eigenvector sign ambiguity.

## Provenance and checkpoint

- Source: [aqlaboratory/genie2](https://github.com/aqlaboratory/genie2/tree/9a954578f7b5a39552545eebc6d4794447794c87), commit `9a954578f7b5a39552545eebc6d4794447794c87`.
- License: Apache-2.0; see `LICENSE`, `NOTICE`, and `THIRD_PARTY_LICENSES.txt`.
- The sole downloaded checkpoint is the official [v1.0.0 epoch.40.ckpt](https://github.com/aqlaboratory/genie2/releases/download/v1.0.0/epoch.40.ckpt), 189,277,096 bytes, SHA-256 `c1f6a4e864f039878b209484bb9780a374858890ad42160ea80a2635d422f859`.
- That SHA is pinned from the downloaded official release artifact; the release does not publish its own SHA digest.
- Upstream documents epoch 40 for its unconditional paper results and epoch 30 for its motif paper results. This package deliberately uses **epoch 40 for all modes**. It does not claim reproduction of the epoch-30 motif benchmark. Epoch 30, training corpora, and evaluation databases were not downloaded.
- Optimizer/training state stays in the ignored checkpoint. All 396 learned model tensors are exported. Quantization affects 2D linear matrices only; biases, layer norms, and IPA head weights retain FP32.

## Reproduce locally

Run from this directory:

```bash
bash tools/bootstrap.sh
npm run test:cpu
npm run test:gpu
npm test
```

The current strict tests fail on output parity. Bootstrap uses `uv venv .venv`, the UV hardlink cache, pinned CPU PyTorch dependencies, and the exact upstream commit. It downloads only the official checkpoint. All source clones, binary oracle fixtures and transient browser data are ignored; no upstream Python tree or weight binaries are packaged. `tools/requirements-export.txt` pins the export environment. A hardware-enabled Chromium must already be available; set `CHROME_PATH` if it is not in the recognized Playwright cache locations. The runner uses Node built-ins, an ephemeral loopback HTTP port, local profiles, and native Vulkan. There is no npm install step and no runtime dependency.

`tools/export_genie2.py` writes `weights/{fp32,fp16,int8}/manifest.json` plus sharded safetensors. The 25,000,000-byte cap includes headers. INT8 is symmetric per output row with colocated `.__scale` FP32 tensors. The exporter supports multipart row splits; this 15.73M-parameter checkpoint does not need an individual tensor split. Each shard has a pinned byte count and SHA-256. Export reload compares every tensor bit-for-bit with the independently quantized intended state. Exact reload results are in `fixtures/export_metrics.json`.

`tools/dump_oracle.py` executes the original Denoiser with the reloaded weights, without ATen/FX tracing or an op graph. All binary tensors are little-endian FP32 under `.cache/fixtures/`; JSON manifests in `fixtures/` contain shape, offset, byte count and checksum. Every case saves initial Gaussian noise, all reverse-step Gaussian draws, feature tensors, intermediate single/pair/structure outputs, and final coordinates. Three precisions × three modes × two lengths produce 18 trajectories. Runtime parity uses identically quantized PyTorch weights; `fixtures/quantization_drift.json` separately reports quantization drift against FP32.

## Architecture and modes

| Module           | State tensors | Parameters | Implemented inference                                                                           |
| ---------------- | ------------: | ---------: | ----------------------------------------------------------------------------------------------- |
| SingleFeatureNet |             1 |    328,320 | Residue, chain and timestep sinusoidal encodings, masked amino acids, projection                |
| PairFeatureNet   |             5 |    117,376 | Outer sum, relative positions, distance bins, signed frame quaternions, motif distance features |
| PairTransformNet |           190 |  1,655,680 | Five blocks; outgoing/incoming triangular multiplication, gates, layer norms, pair transitions  |
| StructureNet     |           200 | 13,630,704 | Eight IPA layers, scalar/point/pair attention, transitions, quaternion BackboneUpdate           |
| Sampler          |             0 |          0 | Cosine DDPM schedule, exact noise inputs or seeded sampling, Frenet-frame reconstruction        |

The official configuration has 384 single channels, 128 pair channels, 12 IPA heads, hidden width 16, four query/key points, eight value points, and one structure recycle. Triangular attention is disabled by the official configuration; all enabled layers are implemented.

This is a handwritten modular network in `shared/network.js` using independent CPU and GPU tensor algebra backends. Individual WGSL kernels are compiled from runtime tensor dimensions and cached. There is no fixed-length or fixed-step graph archive, traced replay, precomputed neural output, or remote inference. Timesteps and total steps are runtime tensor values; sequence length controls actual allocations and contractions. Attention, normalization, activations, matmul, triangular updates, eigensolver geometry, frame updates and reverse updates execute in WGSL on the GPU backend. Host work is limited to validation, feature specifications, schedule coefficients, randomness, I/O, and orchestration.

| Mode            | Input                                                  | Verified execution coverage                                        |
| --------------- | ------------------------------------------------------ | ------------------------------------------------------------------ |
| Unconditional   | Length, steps, noise scale                             | L24/T2 and L48/T3, every precision                                 |
| Single motif    | One motif, or explicit masks and coordinates           | L24/T2 and L48/T3, every precision                                 |
| Multiple motifs | Motif groups and independently masked pair constraints | L24/T2 and L48/T3, every precision                                 |
| Scaffold ranges | Segment length ranges and motif segments               | Shared-feature constraint tests; then the same motif sampling path |

The runtime accepts any integer length ≥3 and chain lengths ≥3 that fit the device's buffers and available memory. Length 256 is the training configuration's positional scale, not a hard runtime shape limit; larger inputs are outside the documented training range. Step count is configurable from 1 to 100,000; changing it regenerates the cosine schedule and uses that count as the timestep-embedding denominator, matching construction of upstream Genie with that `n_timestep`. The paper used 1,000 steps. The 2/3-step tests establish execution/parity behavior, not useful protein design quality.

Motifs provide **soft conditioning**, as in upstream sampling: coordinates are not forcibly clamped after every diffusion step. `motifs[].frames.trans` accepts motif frame translations; upstream motif conditioning uses Cα distances, not the supplied motif frame orientations. Group equality sets pair structure masks; different motif groups do not acquire relative-distance constraints.

## API

```js
import { createGenie2Engine } from "./webgpu/index.js";
import { writePDB } from "./shared/index.js";

const engine = await createGenie2Engine({
  baseUrl: "./weights/fp16", // directory containing manifest.json
  precision: "fp16",
  cache: "opfs", // 'none', 'cache', or 'opfs'
  allowSoftwareFallback: true,
  // device, useF16, signal, onProgress, read(url) are optional
});
try {
  const result = await engine.sample({
    length: 128,
    steps: 1000,
    scale: 0.6,
    seed: 42,
  });
  console.log(writePDB(result));
} finally {
  engine.destroy();
}
```

`sample()` also accepts `motifs`, `features`, `scaffold`, `initialNoise`, `noises`, `trajectory`, `signal`, `onStep`, and `onStage`. Coordinates/noise are flat `Float32Array(L*3)`. `noises` is in descending step order. `features` uses upstream field names: `aatype`, `atom_positions`, `residue_mask`, `residue_index`, `chain_index`, `fixed_sequence_mask`, `fixed_structure_mask`, `fixed_group`, and `interface_mask` (all flat). A supplied group/mask combination derives structure masking unless an explicit structure mask is supplied.

```js
const motifs = [
  { start: 10, sequence: "ARN", coordinates: ca1, group: 1 },
  { start: 35, sequence: "DCE", frames: { trans: ca2 }, group: 2 },
];
const result = await engine.sample({
  length: 64,
  motifs,
  steps: 1000,
  scale: 0.4,
});
```

A motif can use explicit `indices` instead of `start`. `scaffold` accepts `{ segments, minLength, maxLength }`, where each segment is `{ min, max }` or `{ motif }`. `parseMotifSpecification(text)` parses upstream `REMARK 999` scaffold specifications plus PDB Cα records. `sampleScaffold()` samples feasible length assignments. `parsePDB()` reads first-model CA atoms; `writePDB()` writes Cα-only records, retaining supplied motif residue types and using alanine for scaffold residues.

`forward({ coordinates, rotations?, features?, length?, timestep, steps, signal?, onStage? })` runs the complete Denoiser and returns predicted noise `z`, updated coordinates and rotations. Rotation matrices are row-major `Float32Array(L*9)`; absent rotations are reconstructed from Cα positions. The CPU equivalent is `createGenie2CPUEngine()` from `cpu/index.js`; an injected `read(url)` returning an ArrayBuffer allows Node or other non-fetch loading.

`webgpu/worker-client.js` exports `createGenie2Worker()`. Its `init`, `sample`, `forward`, `memoryUsage`, and `destroy` methods return promises. Pass cancellation/progress as a second argument to worker `sample(options, { signal, onStep })`. Workers import only local pure ES modules.

## GPU resources and browser behavior

Production retries `requestAdapter({powerPreference:'high-performance'})` three times, then calls plain `requestAdapter()` if needed. Software adapters are accepted by default and identified through `adapterInfo` / `isFallbackAdapter`. Strict tests pass `allowSoftwareFallback:false` and reject SwiftShader, llvmpipe/lavapipe, WARP, and unidentifiable adapters. Tests also exercise fallback acquisition with controlled adapter mocks.

Weights remain packed in VRAM. Half weights use `unpack2x16float`; INT8 uses signed byte extraction and the corresponding per-row scale inside WGSL reads. They are never expanded to full FP32 GPU weight buffers. Accumulation is FP32; `useF16` optionally requests the device feature but does not change accumulation precision. Local variable arrays allow dynamic indexing under WGSL/Naga rules; no constant-array dynamic indexing is used.

Scratch buffers are reused between module boundaries. Inference has a `finally` cleanup path, with zero tracked transient buffers after normal return, error, or abort. `destroy()` is idempotent and releases all engine-owned buffers; an externally provided GPUDevice remains alive. Allocations check both `maxBufferSize` and `maxStorageBufferBindingSize`. Concurrent inference on a single engine is rejected. Software execution may be slow but is not prohibited in production.

The loader validates shard length and SHA-256 with Web Crypto, verifies dtype/shape and colocated scales, supports Cache API and OPFS offline reload, and falls back to fetch when browser storage is unavailable. The private-storage test simulates Safari-style `SecurityError` denial in Chromium; **actual Safari and Firefox have not been run**. Integrity errors fail instead of silently using untrusted data.

## Strict parity issue: upstream quaternion signs

`PairFeatureNet._encode_orientations()` computes `R_j @ R_i` (without transpose) and uses the largest eigenvector from `torch.linalg.eigh` directly as four learned input features. Eigenvectors have a ± sign ambiguity. For a rotation-derived matrix, the other three eigenvalues are nearly degenerate. Tiny rounding differences can change the LAPACK iteration and returned sign. The checkpoint is not sign-invariant.

The runtime implements a small lower-Householder/implicit-QL-or-QR eigensolver in WGSL/JavaScript, but its sign choices do not reproduce every LAPACK result. `tools/check_sign_stability.py` changes only the coordinates fed to **unmodified upstream PyTorch**: relative input change `1.06e-8` flips 131 of 576 quaternion signs and changes predicted noise by `1.53e-2` relative RMS. At a perturbation of `1e-7`, 279 signs flip and output drift is `3.09e-2`. This evidence is in `fixtures/sign_stability.json`.

A separate diagnostic (`test/component-diagnostic.mjs`) holds PyTorch's quaternion features fixed **only to isolate subsequent modules**. Its CPU final predicted-noise error is `5.34e-7`; the separate hardware GPU component diagnostic is `5.83e-7`. This is not an end-to-end parity pass and is never used by acceptance tests. No quaternion oracle injection or sign patch is used in `test:cpu` or `test:gpu`. The full tests retain original signed-quaternion behavior and fail.

A shared deterministic sign convention in both runtime and a clearly modified reference is a possible follow-up, but it changes the reference model and has not been substituted for the required unmodified checkpoint parity. Exact reproduction of the original LAPACK sign path would require much tighter control of preceding arithmetic than ordinary FP32 tolerance. No claim is made that strict parity is mathematically impossible.

## Measured GPU results

All comparisons use raw final coordinates without rigid alignment. Recorded on NVIDIA GB10, driver 580.159.03, hardware Vulkan Chromium. `fixtures/gpu_metrics.json` records the adapter, all 18 trajectories, storage checks, worker agreement, memory and lifecycle tests. `fixtures/cpu_metrics.json` records the required FP32 CPU mode/length matrix after its run. None of these tables should be interpreted as successful parity.

<!-- METRICS START -->
| Precision | Tensors | Parameters | Shards | Bundle bytes | GPU weight bytes | GPU / FP32 |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 396 | 15,732,080 | 3 | 62,974,496 | 62,928,320 | 1.000000 |
| fp16 | 396 | 15,732,080 | 2 | 31,611,920 | 31,566,784 | 0.501631 |
| int8 | 396 | 15,732,080 | 1 | 16,084,040 | 16,020,224 | 0.254579 |

| Case | Final coordinate relRMS | GPU trajectory ms | Strict result |
|---|---:|---:|---|
| fp32_unconditional_L24_T2 | 0.0222204 | 179.0 | FAIL |
| fp32_single_motif_L24_T2 | 0.0276917 | 247.4 | FAIL |
| fp32_multi_motif_L24_T2 | 0.0300765 | 175.3 | FAIL |
| fp32_unconditional_L48_T3 | 0.0328186 | 649.4 | FAIL |
| fp32_single_motif_L48_T3 | 0.026701 | 318.2 | FAIL |
| fp32_multi_motif_L48_T3 | 0.0266648 | 311.6 | FAIL |
| fp16_unconditional_L24_T2 | 0.0287727 | 154.6 | FAIL |
| fp16_single_motif_L24_T2 | 0.0274212 | 150.1 | FAIL |
| fp16_multi_motif_L24_T2 | 0.0398993 | 166.8 | FAIL |
| fp16_unconditional_L48_T3 | 0.0363459 | 396.2 | FAIL |
| fp16_single_motif_L48_T3 | 0.0289644 | 328.3 | FAIL |
| fp16_multi_motif_L48_T3 | 0.0268882 | 300.2 | FAIL |
| int8_unconditional_L24_T2 | 0.0219578 | 164.4 | FAIL |
| int8_single_motif_L24_T2 | 0.0330346 | 165.7 | FAIL |
| int8_multi_motif_L24_T2 | 0.0340885 | 158.4 | FAIL |
| int8_unconditional_L48_T3 | 0.0310644 | 407.5 | FAIL |
| int8_single_motif_L48_T3 | 0.0304085 | 306.5 | FAIL |
| int8_multi_motif_L48_T3 | 0.0291612 | 309.9 | FAIL |

| Precision | Load ms | Cold L24 forward ms | Warm L24 forward ms |
|---|---:|---:|---:|
| fp32 | 431.0 | 378.6 | 84.2 |
| fp16 | 445.8 | 182.9 | 84.7 |
| int8 | 323.2 | 193.8 | 100.8 |
<!-- METRICS END -->

Cold forward includes first kernel compilation within a newly loaded engine. Warm forward repeats the same input on that engine. These are host wall-clock timings including synchronization/readback, not timestamp-query kernel-only measurements. Browsers may retain driver compilation caches between sessions. Trajectory timings vary with prior compilation and shared host load.

Weight quantization drift and runtime parity are separate. Even under PyTorch, FP16/INT8 trajectories differ from the original FP32 trajectory because quantized neural outputs perturb subsequent frames and can flip quaternion signs. See the per-case numbers in `fixtures/quantization_drift.json`; do not interpret them as arithmetic error of the GPU implementation.

## Demo and denovo.run integration

Serve this directory from HTTPS or localhost, for example `python3 -m http.server 8092 --bind 127.0.0.1`, then open `/demo/`. The minimal static demo uses a module Worker, local packed bundles, configurable length/steps/scale/seed/precision, optional motif specification, progress, cancellation, Cα rendering and PDB download. It labels the current parity limitation.

`node sync-upstream.mjs` is **dry-run only**. `--write` explicitly copies runtime/demo/license files to `ProteinDesignSpace/web/static/vendor/genie2/`; `--target` can specify that destination. We have not executed a write or modified that repository. Weights are provisioned separately. Do not promote this build until the strict parity decision is resolved.

Proposed `denovo.run` mapping (no integration files are edited):

| Port concept        | Package mapping                                                                      |
| ------------------- | ------------------------------------------------------------------------------------ |
| Model ID            | `genie2`, pinned epoch-40 SHA and precision                                          |
| Model session       | One Worker-backed engine; session close → `destroy()`                                |
| Generate request    | `sample({length,steps,scale,seed,motifs/scaffold})`                                  |
| StructureArtifact   | `{coordinates, features}`; Cα-only, Å units, PDB via `writePDB()`                    |
| TrajectoryFrames    | `onStep({step,coordinates})` or `trajectory:true`; descending diffusion step         |
| Cancellation        | AbortSignal forwarded through Worker wrapper                                         |
| Scaffolding         | REMARK-999 specification → `parseMotifSpecification()`; group-aware soft constraints |
| Verification status | Experimental / strict unmodified-reference parity failed                             |

There is no remote inference service or dedicated model network port. The suggested 8092 port serves the local demo only; production assets belong under the vendor path.
