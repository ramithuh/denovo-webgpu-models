# SiD-Protein / Distilled 16-Step Proteina Generator

Standalone browser ES modules for the released `16-step-checkpoint.pt`, pinned to [SiD_Protein commit dfef6a29d975451144e8b0449c12cbea28b4ae15](https://github.com/LY-Xie/SiD_Protein/tree/dfef6a29d975451144e8b0449c12cbea28b4ae15), [paper arXiv:2510.03095](https://arxiv.org/abs/2510.03095). The current paper title is SPID: Distilled Protein Backbone Generation. The package name follows the requested SiD-Protein model.

`shared/`, `cpu/`, and `webgpu/` use only local ES-module imports and browser APIs. No npm packages, Node built-ins, inference service, ONNX runtime, or captured execution graph is used by the runtime. The topology is explicit in `shared/model.js`. GPU kernel dimensions and sampler step count are runtime values; changing length or steps does not require an export or trace.

## Reproduce

Run commands from this directory:

```sh
bash tools/bootstrap.sh
npm test
npm run serve
```

Then open `http://127.0.0.1:8080/demo/`. The browser verification harness discovers an existing Chromium executable or uses `CHROME_PATH`. It requires real hardware WebGPU; missing weights, fixtures, or hardware fail the suite. It never skips these checks. `npm run test:cpu` executes the complete FP32 network and sampler in JavaScript. `npm run test:gpu` runs headless Chromium with Vulkan WebGPU and explicitly disables software fallback. Python is only needed for export and oracle regeneration. `uv venv .venv` and hardlinked installs keep disk use low.

The bootstrap downloads only the released 764,630,872-byte checkpoint. Its Hugging Face LFS SHA-256 is `57062dbf5b8a22068b65484b3cee389cd99f1b67bd4e2d280c55366ee22d80e7`. It does not fetch the teacher, evaluation networks, CATH evaluation datasets, or FoldSeek databases. The upstream checkout, environments, browser profiles, checkpoint, bundles, and binary fixtures are ignored by Git.

## Model and sampler audit

The state dict has 490 tensors and 191,102,976 parameters. Configuration is taken from `training/proteina/configs/experiment_config/model/nn/ca_af3_200M_no_tri.yaml` and checked by a strict state-dict reload into the pinned upstream classes.

| Component                 | Configuration                                                                                   |
| ------------------------- | ----------------------------------------------------------------------------------------------- |
| Transformer               | 15 sequential attention + transition layers                                                     |
| Sequence width / heads    | 768 / 12, 64 channels per head                                                                  |
| Conditioning / pair width | 512 / 512                                                                                       |
| Registers                 | 10 learned tokens prepended to residues                                                         |
| Attention                 | Adaptive LayerNorm, node LayerNorm, Q/K LayerNorm, pair bias, sigmoid gate                      |
| Transition                | Adaptive LayerNorm, 4× SwiGLU, adaptive output gate, residual                                   |
| Conditioning              | Time embedding 256 plus three CATH embeddings of 256 each; two 2× SwiGLU projections            |
| Initial sequence features | 128 index sinusoids, 3 zero self-conditioning coordinates, one zero chain-break flag            |
| Pair features             | 127 relative-index bins, 128 zero self-conditioning channels, 64 noisy-coordinate distance bins |
| Pair updates / triangles  | None; the pair representation is fixed through the trunk                                        |
| Output                    | LayerNorm and 768→3 velocity projection, then `x_clean = x_t + (1-t) * velocity`                |

`run_generate_distilled_proteina.sh` selects **16 steps, noise scale 0.45, seed 5**. The Python CLI defaults to 10 steps and 0.4 instead; this package follows the launcher. For `T > 1`, the exact schedule is:

```text
s = round_to_even(linspace(30, 400, T))
t = 1 - 10 ** (-(s / 400) * 2)
```

Every step draws new Gaussian `[L,3]` noise, centers it, multiplies by `noiseScale`, centers it again during interpolation, centers the previous clean prediction, and forms `x_t = (1-t) * noise + t * previous`. The complete denoiser produces the next clean prediction. It does not perform a teacher ODE integration or reuse noise. For `T = 1`, the pinned code instead uses `tInit=0.37` and `onestepGamma=1`; multistep `noiseScale` has no effect. The `T=20` upstream branch calls the same underlying `predict_clean` method and agrees for the all-ones mask used by this sampler.

Although the teacher inference configuration contains `self_cond: true`, this distilled sampling script never passes `x_sc`. Thus self-conditioning features remain zero at every step. Register conditioning and the register rows/columns of the initial pair representation are zero padded, exactly as upstream.

The checkpoint includes C/A/T embedding tables with 6/44/1473 rows. The last rows 5/43/1472 are null embeddings. It does **not** contain the CATH label vocabulary, and the pinned repository expects a separate `cath_label_mapping.pt`. `cathIndices` therefore exposes the actual available conditioning without inventing label mappings. `cathCode` requires the caller's `cathMapping` with the upstream C/A/T dictionaries. Unknown or masked levels use null rows. Empirical length/fold sampling is an application concern; no evaluation datasets are downloaded. Motif and long-chain models added in the pinned repository use other checkpoints and are outside this package.

## Browser API

```js
import { createSidProteinEngine } from "./webgpu/index.js";

const engine = await createSidProteinEngine({
  baseUrl: "./weights/fp16",
  precision: "fp16",
  cache: "cache",
  allowSoftwareFallback: true,
  onProgress: console.log,
});
try {
  const result = await engine.generate({
    length: 64,
    steps: 16,
    noiseScale: 0.45,
    seed: 5,
    // cathIndices: [cRow, aRow, tRow],
    // signal: abortController.signal,
  });
  console.log(result.coordinates, result.pdb);
} finally {
  engine.destroy();
}
```

`generate` returns `{ coordinates, trajectory, times, units: 'nm', pdb }`. All coordinates are `Float32Array`s in the original frame; PDB serialization converts nm to Å and writes Cα-only GLY placeholders. Sequence, sidechains, and other backbone atoms are not predicted.

`forward(coordinates, time, options)` executes every network layer for a single `[L,3]` input. Optional `onTrace(name, Float32Array)` exposes selected module outputs for diagnostics. `generate` accepts `onStep`, `noise` (a flat `[T,L,3]` array of exact **uncentered** Gaussian draws), `tInit`, and `onestepGamma`. Supplied noise enables cross-runtime replay; the dependency-free seeded generator is deterministic but is not PyTorch's RNG.

`createSidProteinCpuEngine` in `cpu/index.js` provides the same forward/generation API. `read(url)` can replace fetch in both loaders. `cache` accepts `'none'`, `'cache'`, or `'opfs'`. Cache failures, including simulated Safari Private Mode storage denial, report a warning and fall back to fetch. Every shard is verified with Web Crypto SHA-256, including cache hits. Multipart row splits and colocated INT8 scales are validated before tensors are assembled.

Production adapter acquisition retries the high-performance preference, then requests an adapter without preferences. Software fallback is allowed by default and exposed through `adapterInfo` and `isFallbackAdapter`. Tests require hardware and reject SwiftShader, llvmpipe/lavapipe, WARP, and reported fallback adapters. Supplying `device` borrows it: engine destruction frees engine buffers but does not destroy the caller's device. A self-managed device is destroyed with its engine. `useF16:false` selects packed-half decoding without requiring `shader-f16`.

`memoryUsage()` reports packed weight bytes, total allocated bytes, live transient buffers, and scratch reuse. Scratch buffers are recycled during a forward and released in `finally`, including errors and aborts. `destroy()` is idempotent. Concurrent calls on an engine are rejected. Length is dynamic within the adapter's storage and dispatch limits; oversized buffers fail with a useful error.

For an off-main-thread session:

```js
import { createSidProteinWorker } from "./webgpu/worker-client.js";
const worker = createSidProteinWorker();
await worker.call("init", {
  baseUrl: new URL("./weights/fp16", location.href).href,
  precision: "fp16",
});
const result = await worker.call(
  "generate",
  { length: 64, steps: 16 },
  {
    signal: abortController.signal,
    onProgress: (kind, event) => console.log(kind, event),
  },
);
await worker.destroy();
```

## Weights and parity

All matrix tensors, including register and fold embeddings, use the selected storage precision; LayerNorm parameters and biases remain FP32. INT8 quantization is symmetric per output row with FP32 scales. FP16 and INT8 remain packed in VRAM and are decoded inside WGSL. Activations and arithmetic use FP32. No full-size dequantized matrix is allocated on the GPU.

| Storage | Model tensors |  Parameters | Safetensors bytes |  Shard cap |
| ------- | ------------: | ----------: | ----------------: | ---------: |
| FP32    |           490 | 191,102,976 |       764,473,768 | 25,000,000 |
| FP16    |           490 | 191,102,976 |       382,713,936 | 25,000,000 |
| INT8    |           490 | 191,102,976 |       192,825,512 | 25,000,000 |

`fixtures/export_metrics.json` records exact per-shard hashes, bytes, and strict tensor reload equality against the intended quantization. `tools/dump_oracle.py` executes the pinned upstream neural classes and original sampler functions. Only evaluation-only imports and Gaussian draw sourcing are isolated; there is no graph tracing. The missing CATH vocabulary is replaced with direct embedding row selection for oracle conditioning. Quantized oracles reload the exported tensors, decode them to FP32, and run the same original PyTorch network.

Binary oracles and exact Gaussian draws live under `.cache/fixtures/`; checked-in JSON includes shapes, sizes, and SHA-256. Tests compare raw, unaligned final coordinates and full trajectories. Quantization drift versus original FP32 is reported separately from runtime parity against identically quantized weights. Categorical distance bins can amplify tiny floating-point differences near cutoffs, so validation covers completed trajectories as well as individual forwards. Diagnostic logs, including early failures, remain in the ignored cache.

<!-- VERIFIED_RESULTS -->

Verified on **NVIDIA GB10, 580.159.03**, 2026-11-22T08:58:07.224Z. Both `npm run test:gpu` and `npm test` passed, with **zero skipped tests**. CPU validation covers all six forward/trajectory cases; GPU validation covers all 18 precision/case combinations plus the original sensitivity regression.

| Precision | Worst GPU forward relRMS | Worst GPU final relRMS | Worst GPU trajectory relRMS | Packed GPU weight bytes |
| --------- | -----------------------: | ---------------------: | --------------------------: | ----------------------: |
| FP32      |                 3.416e-7 |               1.804e-5 |                    1.703e-5 |             764,411,904 |
| FP16      |                 3.893e-7 |               7.337e-6 |                    7.925e-6 |             382,653,952 |
| INT8      |                 3.710e-7 |               6.837e-6 |                    6.480e-6 |             192,740,304 |

Worst CPU final-coordinate relRMS: **2.435e-5**. Packed weight ratios are FP16 **0.500586×** and INT8 **0.252142×** relative to FP32.

| Precision / length | First forward (ms) | Warm forward (ms) | Full 16-step sampling (ms) | Quantization drift vs FP32 relRMS |
| ------------------ | -----------------: | ----------------: | -------------------------: | --------------------------------: |
| FP32 / L32         |              112.3 |              34.6 |                      540.2 |                          0.000e+0 |
| FP32 / L64         |               50.8 |              69.4 |                     1152.9 |                          0.000e+0 |
| FP16 / L32         |               76.8 |              31.3 |                      507.2 |                          1.767e-2 |
| FP16 / L64         |               48.5 |              67.4 |                     1088.3 |                          7.385e-4 |
| INT8 / L32         |               76.7 |              27.7 |                      451.9 |                          2.298e-1 |
| INT8 / L64         |               48.3 |              67.7 |                     1084.1 |                          4.408e-3 |

“First forward” includes shader compilation only for kernels not already compiled by that engine; the first L=32 row is cold, and L=64 reuses the same dynamic pipelines. Timings include queue execution and readback, exclude weight loading, and are measurements on this shared verification host rather than throughput guarantees. Quantization drift describes changed coordinates, not designability. FP32 is the appropriate choice when preserving the original checkpoint trajectory matters.

The original L=64, 8-step, seed-1298 regression now passes at **7.673e-6** final relRMS. The fixes preserve biased-linear accumulation order, PyTorch's Welford and coordinate-sum reductions, SLEEF transcendental arithmetic (including four-lane positional argument reduction), and separate interpolation rounding. No tolerance was relaxed, no generated coordinates are replayed, and no rigid alignment is applied.

Hardware tests also pass for Web Worker forward/full sampling and cancellation; Cache API and OPFS cold/warm reads; simulated Safari Private Mode denial; corrupted-shard rejection; multipart INT8 scale colocation; external GPUDevice ownership; buffer-limit rejection; packed half decoding without shader-f16; and normal/error/abort cleanup. Transient buffers are zero after calls and engine buffer bytes are zero after destruction.

Complete evidence: [GPU metrics](fixtures/gpu_metrics.json), [CPU metrics](fixtures/cpu_metrics.json), [export metrics](fixtures/export_metrics.json), and [validation provenance](fixtures/validation.json). Runtime source SHA-256: `734702fec596e199d1ce37fa1ef0f493aa9cd50494a94d858eeadd63b45cce74`.

<!-- END_VERIFIED_RESULTS -->

## Vendor staging and denovo.run mapping

`node sync-upstream.mjs` previews the runtime/demo files for `ProteinDesignSpace/web/static/vendor/sid-protein/`. `--write` stages them under this package's `.cache/vendor-stage/ProteinDesignSpace/web/static/vendor/sid-protein/`. It intentionally cannot modify another repository during this isolated task. The coordinator can review and promote that tree. Weights and oracle binaries are excluded from vendor source staging and must be served separately alongside their manifest.

| Package operation                                          | Proposed denovo.run port                            |
| ---------------------------------------------------------- | --------------------------------------------------- |
| Worker session / engine                                    | Model session with checkpoint SHA-256 and precision |
| `generate({length, steps, noiseScale, seed, cathIndices})` | Protein-backbone generation node                    |
| `coordinates`                                              | StructureArtifact, Cα atom order, explicit nm units |
| `pdb`                                                      | Downloadable Cα PDB artifact in Å                   |
| `trajectory` / `times`                                     | TrajectoryFrames with flow time and step index      |
| Gaussian `noise`, sampler options                          | Replay metadata attached to the generation artifact |
| AbortSignal / `destroy()`                                  | Cancel node / close model session                   |

No denovo.run application code, sibling model, root configuration, commit, push, or hosting deployment is changed by this package. See `NOTICE`, `LICENSE_SID`, and `LICENSE_PROTEINA` for upstream terms and attribution; Proteina's included license limits use to noncommercial research/evaluation.
