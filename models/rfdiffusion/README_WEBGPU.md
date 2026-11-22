# Original RFdiffusion: browser port

Original RFdiffusion `Base_ckpt.pt`, upstream commit
[`86507b6538f51fce57b5a72477165f03999ed7ae`](https://github.com/RosettaCommons/RFdiffusion/tree/86507b6538f51fce57b5a72477165f03999ed7ae).
This directory contains a standalone ES module implementation of the learned denoiser and a reverse SE(3) sampler. Runtime imports are local modules only. Python, DGL, e3nn, Node and Playwright are development/oracle tools only.

**Validation status is recorded in `fixtures/`. Do not interpret an individual kernel or denoiser check as a release pass.** `npm test` fails if any required fixture, weights, hardware adapter, coordinate comparison, or browser contract is unavailable. The test matrix compares each precision to the same dequantized weights running through the original PyTorch/DGL model. Quantization error against the unquantized model is reported separately.

## Reproduce

From this directory, run `bash tools/bootstrap.sh`. All downloaded source, Python packages, model weights, binaries and browser profiles stay under this directory. The bootstrap pins upstream, installs the versions in `tools/requirements-export.txt`, verifies the checkpoint, exports all three bundles, and regenerates the fixtures. Downloading dependencies requires network access.

```
npm run test:cpu
npm run test:gpu
npm test
```

The hardware harness uses ARM64 Chromium at `CHROME_PATH`, defaulting to `/home/ruh/.cache/ms-playwright/chromium-1243/chrome-linux-arm64/chrome`, and the native Vulkan/ANGLE flags specified for NVIDIA GB10. `DISPLAY` and `WAYLAND_DISPLAY` are removed from the browser environment. Software adapters, undisclosed adapter identities and missing hardware fail. No test skips are permitted. `test/browser.mjs` uses a relative temporary directory to stay inside the workspace without exceeding UNIX socket path limits. Each browser gets its own NVIDIA/Chromium disk cache: a shared cache caused native Vulkan initialization failures during repeated runs on this driver. The harness retries a fresh native process up to five times for Vulkan startup failures only; software adapters, numerical errors, shader errors and missing artifacts fail without a retry.

## Checkpoint and export

Official checkpoint: [IPD Base_ckpt.pt](https://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt).
The downloaded file has 483,616,107 bytes and SHA-256 `0fcf7d7c32b4848030aca3a051e6768de194616f96ba6c38186351a33bfc6eca`. This digest was measured from the official HTTPS download and is pinned in `tools/download_weights.py`.

The checkpoint contains `model_state_dict`, a separate `final_state_dict`, and its training configuration. Inference loads `model_state_dict`, as does upstream. There are **5,998 tensors and 59,808,046 scalar parameters** in that state dictionary. The complete inventory, shapes, dtypes, configuration and per-module counts are in `fixtures/architecture.json`.

`tools/export_rfdiffusion.py` writes `rfdiffusion-safetensors-v1` manifests under `weights/base/{fp32,fp16,int8}/`. All weights are included, including training heads. Each manifest records exact shard byte lengths and SHA-256 digests. The exporter reloads and checks every tensor against the intended quantization, bit for bit. `tools/build_fixtures.py` runs the reloaded bundles through the original model, then `tools/verify_roundtrip.py` independently executes the original checkpoint and writes final-coordinate round-trip metrics to `fixtures/export_metrics.json`.

| Storage | Total shard bytes | Shards | Largest shard |
| --- | ---: | ---: | ---: |
| FP32 | 240,013,816 | 10 | 24,867,320 |
| FP16 | 125,190,104 | 6 | 24,805,976 |
| INT8 | 69,060,952 | 3 | 24,495,464 |

Categorical embeddings, biases, norms, and SE(3) radial/fiber parameters retain F32 storage. Other 2D linear matrices use the selected storage. INT8 uses symmetric per-output-row quantization with a colocated `<name>.__scale` F32 tensor. The browser preserves packed F16/I8 GPU storage, decodes weights in WGSL, and accumulates in F32. `useF16` may request the optional device feature; portable packed-half decoding works without native F16 arithmetic. No Hugging Face upload is performed. `weights/`, `.cache/`, `.venv/` and `node_modules/` are gitignored.

## Architecture audit

The model has 4 extra-MSA blocks, 32 main blocks, and 4 applications of one shared structure refiner. Its dimensions are MSA 256, extra MSA 64, pair 128, template 64, full-graph state 8, and refinement state 64.

| Module | Parameters | Inference work |
| --- | ---: | --- |
| `latent_emb` | 33,536 | 48-channel MSA projection, query/state/pair categorical embeddings, 65-bin relative residue positions |
| `full_emb` | 3,072 | 25-channel extra-MSA projection and query embedding |
| `templ_emb` | 302,144 | Two structure-biased axial pair blocks, torsion projections, pair/state template attention |
| `recycle` | 22,016 | MSA/pair/state normalization, inferred Cβ pair distances, state concatenation and pair projection |
| `simulator` | 59,420,900 | Three-track updates, equivariant attention, radial networks, torsions and rigid frame updates |
| `aa_pred` | 5,397 | 21-class amino-acid logits |
| `lddt_pred` | 3,250 | 50-bin lDDT logits and expected lDDT |
| `c6d_pred` | 16,770 | Exported; bypassed in `return_infer` |
| `exp_pred` | 961 | Exported; bypassed in `return_infer` |

The exact pinned architecture uses **tied axial attention**, rather than AlphaFold triangle-multiplication blocks. MSA-to-pair updates use an outer product. SE(3) blocks use degree-0 scalars and degree-1 vectors, with spherical harmonics through degree 2 and exported e3nn Clebsch–Gordan coefficients. The first convolution is fused by input degree; the final convolution is fused by output degree. Equivariant attention reduces incoming edges at their destination. Group normalization operates on per-degree feature norms. Quaternion frame updates follow the upstream operation order, including individual normalized quaternion products and stored diagonal partial sums. The WGSL arithmetic preserves PyTorch’s F32 division rounding, ARM64 normalization/reduction order, SLEEF exponential polynomial and alphabetic einsum contraction order. DGL scalar edge softmax uses the Arm `expf` table polynomial with two F32 components in WGSL; model tensors and neural accumulators remain F32. Group-normalization inverse variance uses separate scratch passes to preserve the rounding of its double-precision epsilon calculation. SE(3) attention materializes weighted messages before the incoming-edge sum, as the original PyTorch/DGL implementation does. The torsion head preserves upstream in-place ReLU semantics on residual paths.

The inference runner always uses one MSA row and one template. Softmax over either singleton axis is exactly 1, allowing the unused query/key calculation on that axis to be removed without changing its output. MSA row attention and tied pair attention still execute their full learned calculations. This runtime supports the upstream inference input contract, not training with multiple MSA rows or templates.

[`RoseTTAFoldModel.py`](https://github.com/RosettaCommons/RFdiffusion/blob/86507b6538f51fce57b5a72477165f03999ed7ae/rfdiffusion/RoseTTAFoldModel.py) returns from `return_infer` before invoking `c6d_pred` and `exp_pred`. [`model_runners.py`](https://github.com/RosettaCommons/RFdiffusion/blob/86507b6538f51fce57b5a72477165f03999ed7ae/rfdiffusion/inference/model_runners.py) requests that branch and consumes backbone coordinates, torsions, amino-acid logits and lDDT. Self-conditioning supplies the previous backbone prediction as template geometry, with fresh feature recycling state, matching `SelfConditioning.sample_step`. Legacy checkpoint settings `d_time_emb=0` and `d_time_emb_proj=10` are absent from the current constructor; time is carried in `t1d`.

The JavaScript CPU backend uses correctly rounded F32 fused multiply-add for linear maps, tensor contractions, normalization and geometry. This explicitly handles the rare double-rounding cases in `Math.fround(a*b+c)`; native PyTorch midpoint fixtures and an L=64 CPU regression cover the distinction. Generated scalar literals preserve exact F32 values before arithmetic.

The CPU oracle imports the pinned modules and their original DGL graph operators. No DGL replacement is used. CPU-only PyTorch lacks NVTX; only the profiling context manager is replaced with a no-op. DGL 1.1.3, e3nn 0.3.3 and PyTorch 2.9.0+cpu are pinned. Upstream NumPy/SciPy rotation and IGSO(3) tables are retained in oracle generation.

## Runtime API

```js
import { createRFdiffusionEngine } from './webgpu/index.js';
const engine = await createRFdiffusionEngine({
  baseUrl: '/models/rfdiffusion/weights',
  precision: 'int8', // fp32 | fp16 | int8
  cache: 'cache',   // cache | opfs | none
  // device, useF16: false, read: async url => ArrayBuffer,
  // signal, onProgress
});
try {
  const result = await engine.sample({ length: 24, numSteps: 50, seed: 1 });
  console.log(result.pdbText, result.coords, result.trajectory);
} finally {
  engine.destroy();
}
```

`denoiseStep(state, { returnIntermediates, signal, onProgress })` accepts the preprocessed inference state. `shared/preprocess.js` exports `prepareInput({ coords, length, sequence, idx, t, selfCondition, diffusionMask })`. `sequence` uses the checkpoint's 22-class indices, with 21 for the mask token; `diffusionMask` is true for fixed motif residues. The state contains packed arrays `msa_latent [L,48]`, `msa_full [L,25]`, `seq [L,22]`, `xyz [L,3,3]`, `idx [L]`, `t1d [L,22]`, `t2d [L,L,44]`, `xyz_t [L,3,3]`, `alpha_t [L,30]`, and `motif_mask [L]`. Optional recycled features are supported. `state.symmetry='c3'` additionally returns the symmetrized backbone without replacing raw learned outputs.

`denoiseStep` returns raw learned `px0 [L,3,3]`, torsions, sequence logits, lDDT and reconstructed `coords [L,4,3]` in N/CA/C/O order. It returns intermediate traces only on request. The raw amino-acid logits do not turn this structural checkpoint into a sequence-design model.

`sample({ length, contig, inputPdb, symmetry, numSteps, seed, recordedNoise, signal, onProgress })` returns `{ pdbText, coords, sequence, trajectory, timingMs }`. A full schedule is 50 steps. A smaller count runs the prefix beginning at t=50 and returns its last denoised estimate. Contigs support generated ranges (`8-12`), motif ranges (`A10-20`) and chain breaks (`/0 `). `inputPdb` is PDB text. Cn and Dn symmetry require a total residue count divisible by the group order. Each trajectory record holds the timestep, raw learned backbone, reconstructed/aligned `px0`, reverse-step coordinates and IGSO(3) rotation update. Original RFdiffusion reverse steps contain N/CA/C coordinates; newly generated oxygen/sidechain positions are undefined until all-atom reconstruction, while motif sidechains are retained. Returned final backbone/PDB coordinates include O.

For deterministic comparisons, `recordedNoise` can supply `initialCoords`, `sequence`, `diffusionMask`, `idx`, and `steps: [{ translation: Float32Array(L*3), rotation: Float32Array(L*3) }]`. The runtime recomputes the IGSO(3) score and exponential map from the recorded Gaussian variates. It does not replay predicted coordinates. `shared/diffusion.js` also exports the lower-level reverse update.

Motif alignment uses Horn's quaternion solution with Jacobi diagonalization and applies the resulting transform in WGSL. The reverse update holds motif positions fixed. The Base checkpoint's `freeze_track_motif=false` means raw learned motif predictions themselves can move; this matches upstream and is distinct from the fixed coordinates passed to the next reverse step.

All neural layers, spherical-harmonic bases, equivariant tensor products, frame updates, motif alignment, symmetry and reverse coordinate updates execute WGSL in the GPU engine. Parsing and model-input feature construction use ordinary JavaScript. Native Cache API and OPFS caches verify shard hashes on read; storage denial falls back to fetching. Web Crypto is mandatory. One engine serializes inference; concurrent operations on it fail. Scratch allocations are reused across blocks and released in `finally`. `memoryUsage()` exposes tracked GPU bytes and transient buffers. `destroy()` is idempotent, releases model buffers, and destroys a device only when the engine created it.

## Verification artifacts

`fixtures/mode_matrix.json` enumerates 12 single-step cases (three precisions, three modes, plus unconditional self-conditioning) and nine three-step trajectories. Every single-step case checks embeddings, recycling, template blocks, each simulator/refiner block's MSA/pair/state/rotation/translation/torsion output, and final backbone coordinates. Additional unconditional L=64 checks cover every GPU precision and the FP32 CPU backend. Binary traces live only in `.cache/fixtures/`; JSON metadata records their size and SHA-256, which the tests verify before comparison. Trajectory Gaussian variates are independent draws from a stream seeded once with 444; preparing the next model input does not reseed it. Older repeated-noise development fixtures are retained locally under `.cache/fixtures/repeated-noise-diagnostics/`. Symmetric self-conditioning follows the upstream runner’s second all-atom reconstruction before the reverse update. Export fixtures use identical dequantized F32 arithmetic in PyTorch for FP16 and INT8 storage.

The measured final-output matrix, quantization RMSD and hardware timings are appended below by `tools/render_metrics.py`. A failing check is a release blocker.

## Demo and vendor integration

Serve this directory using any static HTTP server, then open `demo/`. The demo uses a real ES module worker, supports the three modes and precisions, shows step progress and trajectory frames, and provides cancellation, PDB download and GPU release.

`node sync-upstream.mjs TARGET` previews the vendor list; add `--write` to copy it. The allowlist includes only `shared/`, `cpu/`, `webgpu/`, `LICENSE` and this README. It excludes development tools, tests, weights, fixtures, caches and the demo. The reproducible smoke destination is `.cache/vendor-smoke`.

For `denovo.run`, map `{ coords, sequence, pdbText }` to a `StructureArtifact` with Angstrom units, backbone atom order N/CA/C/O, the pinned checkpoint/precision, generation mode, contig and seed. Map each `trajectory[k].px0` and `.coords` to `TrajectoryFrames` with explicit atom order and timestep; `.coords` is the N/CA/C reverse state and `.px0` is the reconstructed N/CA/C/O estimate. Store raw `xyz` and `rotation` as optional reproducibility data. The engine lifetime belongs to a worker-backed model session. Cancellation forwards an AbortSignal, and closing the session calls `destroy()`.

<!-- measured-metrics -->
## Measured validation

These measurements compare actual learned outputs against the pinned upstream PyTorch/DGL runner. All coordinate errors use the original coordinate frame, without fitting a rigid transform. The mode matrix uses L=24, with additional unconditional L=64 checks for all precisions; each trajectory contains t=50, 49, 48. Quantized oracles use the identical exported weights dequantized to F32.

| Precision / mode | CPU final xyz relRMS | GPU final xyz relRMS | GPU N/CA/C/O relRMS | Worst GPU block trace |
| --- | ---: | ---: | ---: | ---: |
| fp32 / unconditional | 0.000e+00 | 0.000e+00 | 2.144e-08 | 1.438e-07 |
| fp32 / unconditional (self-conditioned) | 0.000e+00 | 0.000e+00 | 2.100e-08 | 1.589e-07 |
| fp32 / motif | 0.000e+00 | 0.000e+00 | 2.595e-08 | 1.284e-07 |
| fp32 / symmetric | 0.000e+00 | 0.000e+00 | 3.141e-08 | 2.298e-07 |
| fp16 / unconditional | 0.000e+00 | 0.000e+00 | 2.492e-08 | 2.520e-07 |
| fp16 / unconditional (self-conditioned) | 0.000e+00 | 0.000e+00 | 2.466e-08 | 1.663e-07 |
| fp16 / motif | 0.000e+00 | 0.000e+00 | 1.456e-08 | 1.611e-07 |
| fp16 / symmetric | 0.000e+00 | 0.000e+00 | 2.914e-08 | 1.890e-07 |
| int8 / unconditional | 0.000e+00 | 0.000e+00 | 2.160e-08 | 1.602e-07 |
| int8 / unconditional (self-conditioned) | 0.000e+00 | 0.000e+00 | 2.460e-08 | 1.690e-07 |
| int8 / motif | 0.000e+00 | 0.000e+00 | 2.220e-08 | 1.579e-07 |
| int8 / symmetric | 0.000e+00 | 0.000e+00 | 2.093e-08 | 2.041e-07 |

| Three-step trajectory | CPU px0 relRMS | GPU px0 relRMS | GPU raw xyz relRMS | GPU reverse N/CA/C relRMS | GPU rotation relRMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp32_unconditional | 3.978e-07 | 4.932e-07 | 4.907e-07 | 6.803e-08 | 1.522e-07 |
| fp32_motif | 4.147e-07 | 9.545e-07 | 6.423e-07 | 5.237e-08 | 1.438e-07 |
| fp32_symmetric | 3.601e-06 | 4.408e-06 | 3.796e-06 | 9.265e-08 | 1.549e-07 |
| fp16_unconditional | 9.002e-07 | 1.405e-06 | 1.430e-06 | 7.461e-08 | 1.701e-07 |
| fp16_motif | 8.038e-07 | 7.642e-07 | 1.371e-06 | 4.979e-08 | 1.454e-07 |
| fp16_symmetric | 1.432e-06 | 2.544e-06 | 2.117e-06 | 7.354e-08 | 1.657e-07 |
| int8_unconditional | 7.003e-07 | 6.678e-07 | 6.379e-07 | 6.678e-08 | 1.663e-07 |
| int8_motif | 6.257e-07 | 5.480e-07 | 4.863e-07 | 5.555e-08 | 1.562e-07 |
| int8_symmetric | 1.814e-06 | 1.521e-06 | 2.405e-06 | 6.734e-08 | 1.498e-07 |

| L=64 precision | CPU final xyz relRMS | GPU final xyz relRMS | GPU N/CA/C/O relRMS | Worst block trace |
| --- | ---: | ---: | ---: | ---: |
| fp32 | 0.000e+00 | 0.000e+00 | 2.226e-08 | 2.337e-07 |
| fp16 | — | 0.000e+00 | 2.094e-08 | 2.729e-07 |
| int8 | — | 0.000e+00 | 2.219e-08 | 2.915e-07 |

| Export precision | Worst unaligned Cα RMSD vs original FP32 (Å) |
| --- | ---: |
| fp32 | 0.000000000 |
| fp16 | 0.003371716 |
| int8 | 0.172059121 |

| Third-step estimate / mode | FP16 Cα RMSD vs FP32 (Å) | INT8 Cα RMSD vs FP32 (Å) |
| --- | ---: | ---: |
| unconditional | 0.003260756 | 0.088935084 |
| motif | 0.007297685 | 0.495922535 |
| symmetric | 0.011880554 | 0.171122678 |

These three-step quantization comparisons use the same initial coordinates and recorded Gaussian variates, verified independently. They measure model quantization drift, separately from browser-versus-PyTorch implementation parity.

| NVIDIA GB10 precision / length | Cold bundle/device load (ms) | Median warm denoiser step (ms), 3 runs |
| --- | ---: | ---: |
| fp32 / L=24 | 1890.4 | 233.2 |
| fp32 / L=64 | 1890.4 | 825.2 |
| fp16 / L=24 | 1103.6 | 263.8 |
| fp16 / L=64 | 1103.6 | 794.6 |
| int8 / L=24 | 842.2 | 259.4 |
| int8 / L=64 | 842.2 | 863.3 |

Cold loading uses local HTTP with application caching disabled; it includes hashing, parsing, upload and device creation. Warm timings follow one untimed forward pass and include final-coordinate readback. They exclude optional intermediate trace readback. GPU and CPU timings depend on other activity on this shared host.

Browser contract tests confirm caller-owned GPU device survival after engine destruction, module-worker inference, verified offline Cache API and OPFS reads, and a storage-denied fallback matching Safari Private Mode behavior. This fallback is tested by injecting the browser storage error; Safari itself was not run. Buffer checks require zero transient allocations after normal inference, an injected exception and an AbortSignal, and zero tracked bytes after idempotent destruction.

| Final validation command | Passed | Failed / skipped | Wall time (s) |
| --- | ---: | ---: | ---: |
| `npm test` | 12 | 0 / 0 | 651.350 |
| `npm run test:cpu` | 1 | 0 / 0 | 561.227 |
| `npm run test:gpu` | 2 | 0 / 0 | 42.350 |
| `node --test test/benchmark.test.mjs` | 1 | 0 / 0 | 28.305 |

Recorded 2026-11-22T04:17:21.589Z. Runtime source SHA-256: `b499dab4633b9dcb4e33f3ce5170c3f785a111655866182aab059140fe50f783`. The CPU suite uses independent Node workers for fixture batches; runtime modules themselves import no Node APIs. Full logs are saved locally under `.cache/final-*.log` and their hashes are in `fixtures/validation.json`.

