# Proteina v1.2 · 200M no-triangle · WebGPU

A standalone, zero-runtime-dependency ES module port of **one checkpoint**:
`proteina_v1.2_DFS_200M_notri.ckpt`. The full neural forward and iterative flow
sampler run locally. Neural operators execute as WebGPU WGSL compute shaders.
The separate JavaScript CPU implementation is a reference backend.

Upstream: <https://github.com/NVIDIA-BioNeMo/proteina>, pinned to
`a44b407daf6a5358e43cd68907f3e3f1cbc65fdc`.
Checkpoint SHA-256:
`b5f4c1457fac0e7f54b909245b36b95ddd6bfd314ddd51973a202a3a5a2b6ce9`.

**License correction:** this pinned commit's top-level NVIDIA license includes
section 3.3 restricting use to research or evaluation. It is not a blanket
Apache-2.0 release. See [LICENSE.NVIDIA](LICENSE.NVIDIA) and the preserved
[component notices](NOTICE.md). No commercial-use license is implied here.

## Use

Serve this directory over localhost or HTTPS. Weight bundles live under
`weights/{fp32,fp16,int8}/manifest.json`; every referenced shard must be served
beside its manifest, with CORS enabled if hosted on a different origin.

```js
import { createProteinaEngine } from './webgpu/index.js';
import { toPDB } from './shared/index.js';

const engine = await createProteinaEngine({
  baseUrl: new URL('./weights', location.href).href,
  precision: 'fp16',
  cache: 'cache', // 'none', 'cache', or 'opfs'
  // allowSoftwareFallback: true is the production default
});
try {
  const result = await engine.sample({
    length: 64, steps: 400, seed: 5,
    schedule: 'log', schedulePower: 2,
    mode: 'sc', noiseScale: 0.45, selfCondition: true,
    onProgress: ({ step, steps }) => console.log(step, steps),
  });
  const pdb = toPDB(result.coordinates); // input nm; PDB Å
  console.log(pdb, engine.adapterInfo, engine.memoryUsage());
} finally {
  engine.destroy(); // idempotent
}
```

`forward(Float32Array(L * 3), { t, xSC?, fold?, residueIndices?, chainBreaks?,
signal?, trace? })` returns `{coordinates, length, units: 'nm', timingMs}`.
These coordinates are the **network velocity prediction**, matching upstream
`coors_pred`, not clean positions. `sample()` returns the final backbone positions.
Time must be in `[0, 1)`. Missing self-conditioning is distinct from an explicitly
provided all-zero self-conditioning array because their distance features differ.

`createProteinaEngine` additionally accepts `device`, `useF16`, `read(url)`,
`signal`, `onProgress`, and `allowSoftwareFallback`. An external device remains
owned by its caller. Strict hardware mode requires usable `device.adapterInfo`
for externally supplied devices. `useF16: true` requests native `shader-f16`
loads; the default unpacks the same packed FP16 words in WGSL without requiring
that feature. Accumulations and activations remain FP32 in both modes.

The engine retries high-performance adapter acquisition three times and then
tries the default adapter request. It exposes `adapterInfo` and
`isFallbackAdapter`; production allows a software adapter. Tests set
`allowSoftwareFallback: false` and reject SwiftShader, llvmpipe, lavapipe and WARP.

`createProteinaCPU` in `cpu/index.js` exposes the same `forward` and `sample`
interfaces and supports an injected `read` for Node tests. It decodes weights to
host FP32, uses double-precision dot-product accumulators, and stores activation
results in FP32 arrays. It never participates in WebGPU inference.

For a Web Worker:

```js
import { createProteinaWorker } from './webgpu/worker-client.js';
const worker = createProteinaWorker();
await worker.init({baseUrl: new URL('./weights', location.href).href,
                   precision: 'int8', cache: 'opfs'});
const result = await worker.sample({length: 64, steps: 100},
                                  {signal, onProgress});
await worker.destroy();
```

Workers transfer final coordinate buffers. A worker operation can be cancelled
with an AbortSignal. Each engine serializes its operations by rejecting concurrent
calls. Abort/error paths release transient buffers in `finally` blocks.

## Architecture audit and mode matrix

The checkpoint has **191,102,976 parameters in 490 tensors**. Its saved config
has token width 768, pair width 512, conditioning width 512, 15 transformer
blocks, 12 heads of width 64, and 10 learned register tokens. The network consumes
coordinates in nanometers and predicts velocity. There are no triangle modules,
pair updates, distogram head, sequence decoder, or side-chain decoder in these
weights.

`shared/network.js` explicitly composes the original architecture: sinusoidal
residue/time features; C/A/T embeddings; coordinate and sequence projections;
two conditioning SwiGLU transitions; binned pair distances and relative residue
separation; pair adaptive layer normalization; padded registers; all 15 blocks
of adaptive normalization, full-width Q/K normalization, pair-biased gated
attention and SwiGLU transitions; register removal; final layer norm and 3D head.

Only residue-invariant conditioning is computed once and broadcast. Pair
one-hot projection uses an equivalent sparse column sum. Trace tests expand
broadcast conditioning and compare all 15 blocks with original PyTorch outputs.
The executable network is hand-written and dynamic; there is no ATen/FX capture,
recorded operation list, fixed-L shader, or fixed-step sampler.

| Checkpoint mode | Package behavior / verification |
| --- | --- |
| Unconditional Cα generation | ODE and SDE, L=32/64, multi-step final-coordinate parity |
| Self-conditioning on/off | Denoiser both ways; self-conditioned trajectories and guided trajectory with it disabled |
| CATH fold conditioning | Present: embeddings C=6, A=44, T=1473, including null rows 5/43/1472; explicit `fold: [c,a,t]` |
| CATH strings | `foldIndices(code, mapping)` requires caller-supplied official vocabulary; checkpoint contains none |
| Classifier-free guidance | Same checkpoint, `guidanceWeight`, requires fold indices; tested at 1.6 |
| Schedules | `uniform` (`linear` alias), `log`, `power`, `cos_sch_v_snr`, `loglinear`, `edm`; original scalar fixtures at 3/7/400 steps; 400-step scalar simulation tests the final ODE cutoff |
| SDE g(t) | `1/t`, `us`, `tan`, power transformation and optional clamp, all scalar parity checked |
| Noise modulation | `noiseScale` multiplies variance as upstream; exact saved draws accepted via `noise` |
| Score modulation | `scoreScale` validated but unused, matching the pinned upstream implementation |
| Final ODE switch | Switch after t > 0.99; after t > 0.985 for cosine/EDM schedules |
| Coordinate processing | Center initial noise, center predicted-clean/current coordinates for velocity, and center each update; final PDB conversion nm → Å |
| Residue gaps/chain breaks | Optional explicit residue indices and per-residue chain-break features; one unpadded protein per invocation |
| Batches | Invoke the engine sequentially for individual proteins; padded batch tensors are not a public API |
| Autoguidance / LoRA / motif scaffolding | Require another checkpoint, extra weights or separately trained features; not modes of this standalone weight bundle |
| Sequence design / all-atom relaxation / designability | Outside this Cα model; no evaluation weights or databases downloaded |

Use `noise: [initialGaussian, ...sdeGaussianDraws]` for cross-framework replay.
Each draw is an uncentered `Float32Array(L * 3)`; the sampler applies upstream
centering. `seed` uses a documented local deterministic generator and does not
claim the same random stream as `torch.manual_seed`. `trajectory: true` retains
all updated positions. Test step counts intentionally exercise numerical parity,
not biological sample quality; normal generation uses many more steps.

## Export and reproduce

From this directory:

```sh
mkdir -p .cache weights
# Skip clone if already present.
git clone --no-checkout https://github.com/NVIDIA-BioNeMo/proteina.git .cache/upstream
git -C .cache/upstream checkout a44b407daf6a5358e43cd68907f3e3f1cbc65fdc
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python --link-mode hardlink \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match -r tools/requirements-export.txt
.venv/bin/python tools/download_weights.py
.venv/bin/python tools/export_proteina.py
.venv/bin/python tools/dump_oracle.py
.venv/bin/python tools/dump_sampler.py
npm test
npm run serve
```

The downloader fetches only the specified NGC checkpoint and verifies SHA-256
before publishing the file. Export preserves row-major tensor layout, splits
oversized tensors on complete output rows, and colocates each INT8 part's
`.__scale` vector. Each shard, including metadata, is at most 25,000,000 bytes.
Every tensor is reloaded, checked exactly against its quantized representation,
and loaded strictly into the upstream network. Norm/bias vectors, registers and
fold embeddings remain FP32; dense projection matrices are FP32, FP16 or symmetric
per-output-row INT8. No optimizer tensors are exported.

The oracle executes the original upstream neural modules and original
`R3NFlowMatcher.full_simulation`, with explicit clean-prediction/CFG plumbing
matching `ModelTrainerBase`. Compatibility shims restore the removed `torch.Dict`
typing alias and provide a pure Torch `scatter_mean` implementation. The latter
is not called by the checkpoint's `sample` multilabel mode. Synthetic four-part
CATH labels map to known embedding indices **only for fixtures**; they are not
represented as official fold names. No neural modules are replaced.

Large oracle binaries remain in `.cache/fixtures/`; metadata and checksums are
in `fixtures/`. Missing weights, missing fixtures, invalid checksums or absent
hardware fail tests rather than skipping them. `CHROME_PATH` can select the
Chromium binary. The browser harness uses NVIDIA Vulkan hardware on the
verification host, isolated profiles inside `.cache/`, and no npm dependencies.

## Measured verification

<!-- METRICS_START -->
| Precision | Tensors | Shards | Bundle bytes | Largest shard | Resident weights + constants |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp32 | 490 | 46 | 764,469,552 | 23,616,944 | 764,413,436 |
| fp16 | 490 | 16 | 383,506,160 | 24,846,912 | 383,450,620 |
| int8 | 490 | 8 | 194,007,924 | 24,638,320 | 193,928,408 |

Export round-trip maximum absolute error: **0 for all precisions**, with strict upstream model reload.

Hardware: **NVIDIA GB10, 580.159.03, GPU-cb4201ed-0d40-feb9-d1ff-5766b26ed9a1**. Browser: `Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/153.0.0.0 Safari/537.36`.

| Backend | Worst denoiser relRMS | Worst final sampler relRMS | Warm L=32 / L=64 (ms) |
| --- | ---: | ---: | --- |
| CPU FP32 | 5.646e-7 | 3.155e-5 | reference backend |
| GPU fp32 | 5.906e-7 | 3.154e-5 | 42.6 / 54.2 |
| GPU fp16 | 6.061e-7 | 4.004e-6 | 30.6 / 41.8 |
| GPU int8 | 7.142e-7 | 3.797e-6 | 37.4 / 48.0 |

Every final tensor passes **relRMS < 1e-4**, with **zero skipped tests**. Cases cover L=32/64, T=2/3/4, ODE/SDE, log/linear/power schedules, self-conditioning on/off, fold conditioning and CFG.

| Quantized PyTorch vs FP32 PyTorch | Worst denoiser drift | Worst final sampler drift |
| --- | ---: | ---: |
| fp16 | 6.991e-5 | 1.580e-3 |
| int8 | 4.541e-3 | 1.136e-2 |

Resident weight ratios: FP16 **0.5016×**, INT8 **0.2537×** FP32. Native-F16 load parity: **4.816e-7**.

| Precision | Cold load (ms) | First forward including compilation and traces (ms) |
| --- | ---: | ---: |
| fp32 | 2265.5 | 82.4 |
| fp16 | 1158.6 | 52.6 |
| int8 | 690.9 | 80.0 |

Transient buffers after normal/error/abort paths: **0**. Tracked bytes after idempotent destroy: **0**. Offline Cache API and OPFS worker runs, both simulated Private Mode fallbacks, and the interactive demo pass. Warm numbers are medians of three full forward calls and include dispatch/readback overhead. Cold loads use the local HTTP fixture server.
<!-- METRICS_END -->

All GPU precision comparisons are against **identically quantized PyTorch**.
Quantization drift against the unquantized model is reported separately and is
not mislabeled as runtime parity. The test suite includes real worker inference,
offline Cache API and OPFS reloads, simulated Safari Private Mode storage errors,
input/submit/allocation/callback failures, live abort, initialization abort,
limits, repeat execution, native-F16 loads, and demo interactions. Safari itself
and Firefox are not hardware-tested here. WGSL uses standard storage/uniform
buffers and no dynamically indexed constant arrays.

## Memory and deployment

`memoryUsage()` reports tracked GPU buffer allocations, including packed weights,
scratch, transient bytes and transient buffer count. It does not estimate driver
pipeline-cache memory. Weights stay packed in VRAM; matmul shaders decode values
on demand. Scratch buffers are reused by size, with idle retention capped at
128 MiB after each operation. All tracked allocations reach zero after destroy.

The largest pair buffer is `(L + 10)^2 * 512 * 4` bytes. Length is dynamic within
the adapter's `maxBufferSize` and `maxStorageBufferBindingSize`, checked before
allocation; individual matrices are also checked. Large linear dispatches are
split across workgroup dimensions. This full-attention model has quadratic pair
memory and is not a claim of unlimited browser sequence length.

Proposed **denovo.run port mapping** (no production integration performed):

| denovo.run field / path | Package mapping |
| --- | --- |
| Model ID | `proteina-v1.2-200m-notri` |
| Import | `/static/vendor/proteina/webgpu/index.js` |
| Worker | `/static/vendor/proteina/webgpu/worker.js` |
| Weight root | Configured `baseUrl` with `fp32/`, `fp16/`, `int8/`; upload pending coordinator review |
| Length / steps | `sample({length, steps})` |
| Temperature / solver | `noiseScale` / `mode: 'sc' or 'vf'` |
| Fold | Explicit indices or `foldIndices(code, officialMapping)` |
| Result | `Float32Array` Cα positions in nm; `toPDB` emits Å |
| Local demo port | `8096`, override with `PORT` |

`node sync-upstream.mjs /path/to/ProteinDesignSpace` prints a dry run targeting
`ProteinDesignSpace/web/static/vendor/proteina/`. The coordinator may add
`--write` when ready. It copies only runtime/demo/docs/notices, not cached
upstream code, checkpoints, fixtures, or virtual environments. Nothing has been
committed, pushed, uploaded, or copied to the production vendor destination.
