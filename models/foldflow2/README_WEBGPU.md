# FoldFlow 2 — standalone WebGPU

FoldFlow 2 ReFT backbone generation with a handwritten, modular ES-module neural runtime. There are no npm runtime dependencies, Node imports in runtime code, ONNX graphs, remote inference calls, fixed-length captures, or fixed-step traces. Both CPU and WebGPU execute the sequence encoder, structure encoder, multimodal trunk, structure decoder, torsion head, and flow integration.

**Upstream:** [DreamFold/FoldFlow](https://github.com/DreamFold/FoldFlow/tree/9d2c260813da3c9a2bc944973953f63ea6d71203), commit `9d2c260813da3c9a2bc944973953f63ea6d71203`. Checkpoint: release `0.2.0`, `ff2_reft.pth` (84,544,846 bytes).

**License correction:** the pinned upstream [LICENSE](https://github.com/DreamFold/FoldFlow/blob/9d2c260813da3c9a2bc944973953f63ea6d71203/LICENSE) and README specify **CC BY-NC 4.0**, with attribution to Dreamfold. The request's MIT designation does not match that commit. This package preserves the upstream license in `LICENSE`; ESM, OpenFold, and reused sibling infrastructure notices are in `THIRD_PARTY_LICENSES`. The browser implementation is a modified port, not an upstream release.

## Architecture and mode matrix

| Component | Tensors | Stored values |
|---|---:|---:|
| Structure encoder: 2 IPA / Transformer / edge / frame blocks | 158 | 9,109,976 |
| Structure decoder: 2 blocks, last edge update omitted upstream | 148 | 8,731,864 |
| Sequence adapter and positional embedding | 14 | 293,962 |
| Representation combiner | 12 | 66,688 |
| Multimodal triangular trunk: 2 blocks | 160 | 2,806,016 |
| Decoder adapter | 8 | 83,072 |
| Sequence alphabet mapping buffer | 1 | 22 |
| Frozen ESM-2: 33 layers, 1,280 channels, 20 heads | 564 | 649,401,376 |
| Total | 1,065 | 670,492,976 |

Every IPA block uses eight heads with 256 scalar channels, eight query/key points and twelve value points. Each has a two-layer, four-head, 320-channel Transformer, transition, and quaternion backbone update. The trunk includes gated sequence attention, sequence-to-pair updates, both triangular multiplicative updates, both triangular attention directions, and residual MLPs. Decoder translations use the upstream 0.1 coordinate scaling. Both torsion heads and all FF2 checkpoint tensors are retained; inactive upstream FF2 parameters remain in the export. ESM language-model/contact prediction heads are excluded because FoldFlow consumes only representations and attentions.

| Mode | Inputs | Browser support |
|---|---|---|
| Unconditional generation | `length`, `steps`, optional `seed` or exact `initial` frames | Complete, including frozen ESM-2 and self-conditioning |
| Sequence-conditioned generation | Unpadded `esmTokens`, including BOS=0 and EOS=2 | Complete sequence-to-structure path |
| Precomputed sequence conditioning | `seqEmbed.single`, `seqEmbed.pair` | Complete FF2 structure path; ESM computation bypassed |
| Motif/scaffold/binder, padded batches, multiple chains | Upstream-specific masks and features | Not exposed by this single-chain package |

**Unconditional does require ESM-2.** In `FF2Model._make_seq_mask_pattern`, evaluation sets the mask pattern to **ones**. `FrozenEsmModel` replaces residues with token 32, then runs ESM. Zero mask entries mean unmasked sequence residues, not zero representations. All-mask representations depend on length and are not static constants. ESM token dropout, stored rotary frequency buffers, every transformer layer, all 34 representation levels, all 660 attention channels, and final normalization are implemented. No PDB data, ESMFold weights, or contact-regression checkpoint is downloaded.

Precomputed arrays are contiguous FP32, without batch dimension:

- `single`: `[L, 34, 1280]`, including embedding layer and normalized final layer.
- `pair`: `[L, L, 660]`. The two residue axes follow upstream `attentions.permute(0,4,3,1,2)`, i.e. **key, query**, then layer-major/head-minor channels.
- `seqIndices`: optional integer residue indices (default `1..L`).

## Use

Serve this directory over HTTPS or localhost. The base URL points to a precision directory containing `manifest.json` and its shards.

```js
import { createFoldFlow2Engine } from './webgpu/index.js';
import { toPDB } from './shared/pdb.js';

const engine = await createFoldFlow2Engine({
  baseUrl: './weights/fp16',
  precision: 'fp16',
  cache: 'opfs', // 'none', 'cache', or 'opfs'
  onProgress: console.log,
});
try {
  const result = await engine.sample({ length: 64, steps: 50, seed: 123 });
  const pdb = toPDB(result);
  console.log(engine.adapterInfo, engine.isFallbackAdapter, pdb);
} finally {
  engine.destroy();
}
```

`forward({ length, initial, time=0.6, scCa?, seqEmbed?, esmTokens?, seqIndices?, signal?, trace? })` evaluates one denoiser. `sample` adds `steps=50`, `minTime=0.01`, `selfConditioning=true`, `seed=123`, `rotationScaling=10`, and `onProgress`. `initial` is `[L,7]`, scalar-first quaternion plus xyz translation in Angstroms. Inputs and outputs are flat typed arrays. A numeric seed is deterministic within this package; PyTorch replay uses saved exact initial frames rather than assuming identical random-number generators. Generation is dynamic for positive integer lengths within device capacity and integer step counts **at least 2**, matching the upstream sampler's minimum grid size.

Outputs include predicted `rigids`, `psi`, `rotVectorfield` (`[L,3,3]`), `transVectorfield` (`[L,3]`), and `atoms` (`[L,5,3]`, N/CA/C/CB/O), plus complete padded `atom37` and `atom14` tensors. Sampling additionally returns integrated `frames` and chronological `trajectory`; sampling `atoms` use those final frames and final predicted torsions. Vector fields correspond to the last denoiser input. Quaternion signs are canonicalized to nonnegative scalar component for comparison. The PDB contains all five produced atoms as alanine, matching upstream backbone reconstruction; this model does not design an amino-acid sequence or full sidechains.

The reverse grid is exactly upstream `linspace(min_t,1,T)[::-1]`, with `dt=(1-min_t)/(T-1)`, including the final update at `min_t`. Translation updates are centered after every step. SO(3) updates retain the pinned logarithm's small-angle branch and its `sin(angle/2 + 1e-6)` denominator. The default SO(3) inference multiplier is **10**, from `runner/config/inference.yaml`; a negative multiplier selects the upstream reciprocal-time rule. The unusually coarse T=2/3 oracle runs deliberately preserve upstream's final-step extrapolation. Use the default T=50 for ordinary generation.

All learned operations, SO(3) integration, and atom reconstruction run in WGSL on the GPU backend. Host work handles scheduling, input/positional feature construction, random initial draws, binary I/O, and PDB text. Weights remain packed in VRAM: F16 uses `unpack2x16float`, INT8 uses signed byte extraction and per-row scales in matmul/load shaders. Activations accumulate in F32. The GPU kernels preserve PyTorch’s bias-first fused accumulation and ARM NEON GELU formula; FMA angle reduction avoids native GPU trigonometry errors in ESM rotary embeddings and SO(3) integration. `useF16` controls optional device feature acquisition; it does not convert activations to half precision. All indexed WGSL arrays use function/workgroup storage rather than indexed constant arrays.

`createFoldFlow2CPU({ baseUrl, read? })` exposes the same API with independent JavaScript numerical kernels. `read(url)` must return an `ArrayBuffer`; it can replace browser fetch for local tests. CPU mode decodes weights into host FP32 arrays and is substantially slower than GPU mode. CPU arithmetic kernels use `Function` compilation, so that optional backend requires a CSP permitting it. The GPU backend does not use eval.

Production adapter acquisition retries high-performance adapters three times and then calls plain `requestAdapter()`. Software adapters are accepted by default (`allowSoftwareFallback=true`), with adapter metadata exposed. Tests explicitly set `false` and reject SwiftShader, llvmpipe, lavapipe, WARP, software and unidentified adapters. An externally supplied `device` remains caller-owned. Self-created devices are destroyed by `destroy()`. Calls are serialized; concurrent inference on one engine is rejected. Scratch buffers are recycled at layer/step boundaries and released in `finally`; buffer limits are checked before allocation.

Worker entry: `webgpu/worker.js`, loaded as a module worker. Send `{ id, type:'init', options }`, then `forward` or `sample`; responses are `{id,result}` or `{id,error}`, with `{id,progress}` events. `abort` cancels loading or inference; `memory` and idempotent `destroy` are also supported. Overlapping operations are rejected. `demo/index.html` uses this protocol.

## Reproduce

From this directory:

```sh
bash tools/bootstrap.sh
npm test
```

Bootstrap uses `uv venv .venv`, hardlinked uv packages, the pinned upstream checkout in `.cache/upstream`, and the two verified checkpoints in `weights/`. The lock file records the tested Python 3.12 environment. Python compatibility handling only converts old dataclass defaults to factories. The upstream neural code is unchanged. References were generated with PyTorch 2.15.0 on ARM64. Oracle arithmetic is explicitly FP32 throughout, including ESM; this disables the upstream factory's optional ESM `.half()` cast. FP16 and INT8 oracle models reload the **identically quantized/dequantized exported weights** into PyTorch. Thus parity measures implementation error separately from quantization drift, not equivalence to upstream's mixed-half execution mode.

```sh
.venv/bin/python tools/download_weights.py
.venv/bin/python tools/export_foldflow2.py
.venv/bin/python tools/dump_oracle.py --precision fp32
.venv/bin/python tools/dump_oracle.py --precision fp16
.venv/bin/python tools/dump_oracle.py --precision int8
npm run test:shared
npm run test:cpu
npm run test:gpu
```

Checkpoint SHA-256:

- `ff2_reft.pth`: `73f6d969c6757f4864d19c7a942d698ac5334de3034fcb40a0a9423d2057b902`
- `esm2_t33_650M_UR50D.pt`: `ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c`

`tools/export_foldflow2.py` verifies checkpoint hashes, exports all three precisions, row-splits large matrices, colocates INT8 `__scale` tensors, and strictly reloads every part. Every shard is at most 25,000,000 bytes, including its header. Hashes and shapes are checked again by the browser loader using Web Crypto. Binary oracle fixtures and exact random initial frames live in `.cache/fixtures/`; tracked JSON files contain their shapes and hashes. Missing weights, fixtures or hardware fail tests without skips.

Cache API and OPFS are optional. Read/write failures, including SecurityError and quota/private-browsing denial, fall back to fetched data. Tests verify cold/warm reads and simulated Safari Private Mode storage denial; they do not claim to have run Safari or Firefox. Hardware parity is measured in Chromium with native NVIDIA Vulkan WebGPU.

<!-- VERIFIED_METRICS -->

## Verified measurements

Hardware: NVIDIA GB10. Chromium native Vulkan; no fallback adapter. Full records are in `fixtures/export_metrics.json`, `fixtures/cpu_metrics.json`, and `fixtures/gpu_metrics.json`. `fixtures/verification.json` records the successful command, browser version, and runtime file hashes.

| Precision | Tensors | Safetensors bytes | Shards | Packed GPU bytes | GPU / FP32 |
|---|---:|---:|---:|---:|---:|
| fp32 | 1,065 | 2,682,103,264 | 169 | 2,681,971,904 | 1.000000 |
| fp16 | 1,065 | 1,342,401,952 | 68 | 1,342,281,760 | 0.500483 |
| int8 | 1,065 | 674,400,652 | 34 | 674,233,564 | 0.251395 |

Every exported tensor reloads exactly to the expected quantized representation. Biases, normalization parameters and one-dimensional buffers stay F32. No full-size F32 weight mirror exists in GPU memory.

| Backend / precision | L | Maximum final forward relRMS | Maximum trajectory / final-step relRMS |
|---|---:|---:|---:|
| WebGPU fp32 | 32 | 3.842e-06 | 4.682e-05 |
| WebGPU fp32 | 64 | 3.094e-06 | 8.307e-05 |
| WebGPU fp16 | 32 | 3.858e-06 | 3.359e-05 |
| WebGPU fp16 | 64 | 3.822e-06 | 7.333e-05 |
| WebGPU int8 | 32 | 4.042e-06 | 4.319e-05 |
| WebGPU int8 | 64 | 3.625e-06 | 8.910e-05 |
| CPU FP32 | 32 | 3.915e-06 | 3.974e-05 |
| CPU FP32 | 64 | 3.583e-06 | 7.056e-05 |

149 GPU tensor comparisons and 48 CPU tensor comparisons pass at relRMS < 1e-4, with zero skips. Trajectories use T=2 and T=3 at both lengths. GPU forward and sampler tests execute ESM directly; the CPU sampler reuses its own computed representations as upstream caches them. Conditional token inputs and precomputed representations are checked against an independent sequence-conditioned oracle. Atom14 and atom37 are compared in full, including padding.

CPU inference checks took 15.2 minutes on this host, excluding initial weight loading.

Normal, injected-error and abort paths each leave zero transient GPU buffers. Repeated destroy leaves zero bytes. Tests also pass full worker sampling, cancellation during worker loading, external-device preservation, maximum-buffer admission, Cache API and OPFS cold/warm reads, and both storage-denied fallbacks.

| Precision | L | Cold full forward (ms) | Warm full forward (ms) | Full T=2 sample (ms) | Full T=3 sample (ms) |
|---|---:|---:|---:|---:|---:|
| fp32 | 32 | 761.8 | 155.7 | 609.4 | 274.3 |
| fp32 | 64 | 879.6 | 272.6 | 409.1 | 435.2 |
| fp16 | 32 | 404.3 | 158.2 | 225.3 | 292.2 |
| fp16 | 64 | 506.4 | 283.5 | 709.2 | 430.1 |
| int8 | 32 | 439.9 | 156.5 | 228.7 | 252.1 |
| int8 | 64 | 536.0 | 279.3 | 567.1 | 425.0 |

Times are browser wall-clock durations through GPU completion and readback, including ESM. Cold measurements include shader/pipeline compilation and oracle-comparison trace readbacks; warm measurements use normal outputs. Model download/upload is separate in the JSON. Measurements share the host with other work and are not isolated throughput claims.

| Precision | L | Atom37 quantization drift relRMS | T=2 trajectory drift | T=3 trajectory drift |
|---|---:|---:|---:|---:|
| fp16 | 32 | 3.011e-04 | 1.189e-03 | 3.574e-03 |
| fp16 | 64 | 3.068e-04 | 3.577e-03 | 1.872e-02 |
| int8 | 32 | 2.340e-02 | 6.550e-02 | 1.648e-01 |
| int8 | 64 | 1.940e-02 | 1.336e-01 | 4.095e-01 |

Quantization drift compares quantized **PyTorch** outputs with FP32 PyTorch outputs; it is separate from WebGPU implementation error. Coarse grids amplify that drift. This is numerical verification, not a designability evaluation.

<!-- /VERIFIED_METRICS -->

FP16 is the preferred compact bundle for preserving the FP32 trajectory. INT8 saves more memory but has appreciably larger quantization drift; parity against its own PyTorch oracle does not establish equal design quality.

## Demo and vendor handoff

`npm run serve` serves the demo on `127.0.0.1:8092` (`PORT` overrides it). The intended denovo.run mapping is `/static/vendor/foldflow2/` for runtime modules and `/models/foldflow2/weights/<precision>/` for independently hosted shards. The application model identifier is `foldflow2`; constructor `createFoldFlow2Engine`; output adapter `toPDB`. No remote host, root registry, or coordinator-owned routing file is changed here.

Proposed denovo.run artifact mapping:

| Artifact / operation | Mapping |
|---|---|
| Unconditional request | `length`, `steps`, `seed`, precision, schedule and self-conditioning options |
| `SequenceSet` input | ESM-2 token IDs or supplied ESM representations, with stable sequence/residue IDs |
| `StructureArtifact` output | PDB, atom14/atom37 coordinates and atom masks, quaternion frames, chain A and residue indices |
| `ResidueTrack` output | Psi sine/cosine, translation vector field and SO(3) tangent matrix per residue |
| `Collection` output | Independent seeds/samples sharing resident weights, with chronological frame trajectories |
| Provenance | Pinned commit, both checkpoint hashes, precision, exact initial frames, schedule, steps, elapsed time |

This mapping is an integration proposal; no host application schema is changed.

`node sync-upstream.mjs /path/to/ProteinDesignSpace` prints a dry run targeting `ProteinDesignSpace/web/static/vendor/foldflow2/`. The coordinator may supply `--write` to copy only runtime modules, documentation, and licenses. This task did not execute that write. Weights, fixtures, Python environments, upstream source and test tooling are excluded from vendor sync. All task changes remain inside `models/foldflow2/`; no commit, push, or upload was performed.
