# LigandMPNN WebGPU port

Browser-native port of MIT-licensed `dauparas/LigandMPNN`, based on upstream commit
`26ec57ac976ade5379920dbd43c7f97a91cf82de`. Upstream Python sources remain unchanged.

Architecture: 128 hidden channels, 32 protein neighbours, 25 context atoms,
16 radial basis bins, 16 positional channels, three protein encoders, two paired
context encoders, and three autoregressive decoders. 2,618,693 parameters in 197
tensors. Element identity, periodic group and period form a 147-channel one-hot.
GELU uses the exact erf form; LayerNorm epsilon is 1e-5; message reduction scale
is 30. Context includes non-water heavy atoms (ligands, metals, DNA and RNA).

## Reproducible export and oracle

```sh
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r tools/requirements-export.txt
.venv/bin/python tools/download_weights.py
.venv/bin/python tools/export_ligandmpnn.py
.venv/bin/python tools/dump_oracle.py
.venv/bin/python tools/dump_precision_oracle.py
curl -fL https://files.rcsb.org/download/3HS4.pdb -o .cache/3HS4.pdb
.venv/bin/python tools/dump_small_molecule_oracle.py
```

Exports are `weights/<checkpoint-stem>/<precision>/manifest.json` plus safetensors.
The Genie3 manifest structure is preserved with format `ligandmpnn-safetensors-v1`:
source/checksum, config, parameter counts, shard bytes/SHA256, and per-tensor
shard/dtype/shape. All 2D matrices become F16 or symmetric per-row I8; 1D tensors
remain F32. INT8 `.__scale` F32 companions remain in the same shard. The 25,000,000
byte limit includes the safetensors header. Reload uses strict upstream state-dict
validation. No checkpoints or generated bundles are committed.

| Variant | Precision | Shard bytes | Weight relRMS | Forward logits relRMS | Forward log-probs relRMS |
|---|---|---:|---:|---:|---:|
| 010 | FP32 | 10,493,844 | 0 | 0 | 0 |
| 010 | FP16 | 5,294,796 | 2.020705e-4 | 5.953310e-4 | 1.803667e-4 |
| 010 | INT8 | 2,753,956 | 7.667495e-3 | 2.389686e-2 | 7.880163e-3 |
| 020 | FP32 | 10,493,844 | 0 | 0 | 0 |
| 020 | FP16 | 5,294,796 | 2.030347e-4 | 4.968115e-4 | 1.523157e-4 |
| 020 | INT8 | 2,753,956 | 7.766674e-3 | 2.658461e-2 | 8.250655e-3 |

All rows are single-shard bundles. Errors are measured by reloading weights into
CPU FP32 PyTorch on the same real holo fixture, with teacher-forced decoding.
The requested FP16 logits error <5e-4 against FP32 is already exceeded by 010
weight rounding alone. This is not a WebGPU implementation error; FP16 runtime
parity must also be checked against the identically rounded PyTorch weights.
Full-precision metrics are in `fixtures/export_metrics.json`.

`fixtures/holo_1BC8.pdb` contains the 32 residues closest to the two zinc ions in
upstream 1BC8, retaining DNA, zinc and waters in their original coordinates.
The oracle excludes water and hydrogens, selects the closest 25 context atoms,
and records protein geometry, RBF/projection inputs, features, every encoder,
every autoregressive decoder invocation, logits, probabilities, sequences,
decoding order, and global/ligand negative-log-likelihood scores. It includes
ordinary, tied, fixed-sidechain-context and no-atom-context sampling cases.
The first two residues are fixed, chain C is designed, cysteine is omitted, and
alanine has a +0.2 logit bias. Ties average logits with equal weights.

Fixtures use JSON descriptors with byteOffset/byteLength/shape/dtype and
little-endian F32/I32 binary data. PyTorch multinomial draws and interior CDF
uniforms are recorded for exact portable replay; matching random seeds across
different random number generators is not claimed.

The optional sidechain *packing* model is a separate denoising model depending
on OpenFold torsion/rigid transforms. It is deferred; fixed sidechain *context*
in the sequence model is a separate feature.

## CPU reference and shared API

`npm run test:cpu`: 11/11 passing on NVIDIA GB10 host CPU (ARM64). Worst FP32
intermediate relRMS 2.341330e-6, maximum absolute error at that tensor 7.033348e-6.
Teacher-forced decoder worst relRMS 1.315102e-6. Four exact autoregressive replays:

| Case | Exact sequence | Worst trace relRMS |
|---|---|---:|
| Ordinary | WQLASGHGKNKANRNAAIRGIIYVNYPEILFP | 2.341330e-6 |
| Tied positions | WQLLGGETKNKHNLNAAIRGIIFVNYPEILFP | 2.045118e-6 |
| Fixed sidechain context | WQIGSGNGKNKHNRHAAIRGIVYNKYPEILNP | 1.822127e-6 |
| Atom context disabled | WQIGNGEGIGLGSLLELIAGIIFIRYPDILFP | 1.939242e-6 |

CPU sampling takes approximately 5.5–5.7 seconds for this L=32 fixture. Runtime
code has zero dependencies. Float32Array storage rounds every operation boundary;
CPU dot-product accumulation uses JavaScript doubles. Exact erf GELU is implemented
with a high-accuracy approximation (absolute erf error about 1.5e-7).

```js
import { createLigandMpnnCpuEngine } from './cpu/index.js';
const engine = await createLigandMpnnCpuEngine({ baseUrl: '/weights' });
const result = await engine.design(pdbOrMmcifText, {
  temperature: 0.3, numSequences: 2, seed: 42,
  designChains: ['C'], fixedPositions: ['C8', 'C9'],
  omitAAs: 'C', biasAA: { A: 0.2 },
  tiedPositions: [['C12', 'C29']],
  useAtomContext: true, useSideChainContext: false,
});
engine.destroy();
```

A fixed-position token can identify a residue (including insertion code) or an
entire chain. `redesignedResidues` restricts design to an allowlist. Explicit fixed
positions take precedence. Unknown addresses and overlapping ties raise errors.
Tied groups must share design/fixed status; optional `{residues, weights}` objects
set upstream weighted-logit coupling. Default tie weights average logits.

PDB and mmCIF support the first model, positive occupancy, blank/A/1 alternate
locations, author chain/sequence identifiers, insertion codes, MSE-to-MET (SE to
SD), water exclusion, and all known heavy elements. Missing backbone atoms mask
the residue rather than fabricating coordinates. The 37-atom tensor intentionally
leaves OXT unset, matching upstream `parse_PDB(parse_all_atoms=True)`.

Scores are mean untempered negative log probabilities over designed, valid
positions (global), or their subset within 8 Angstrom of context (ligand), matching
upstream. Fixed-position probability/log-probability output rows are zero.


## WebGPU engine

```js
import { createLigandMpnnEngine } from './webgpu/index.js';
const controller = new AbortController();
const engine = await createLigandMpnnEngine({
  baseUrl: '/weights',
  variant: 'ligandmpnn_v_32_010_25', // or ligandmpnn_v_32_020_25
  precision: 'fp32',               // fp32, fp16, int8
  onProgress: ({ loaded, total }) => console.log(loaded / total),
  signal: controller.signal,
  // device,                       // optional externally owned GPUDevice
  // useF16: false,                 // force core WebGPU packed-half path
});
try {
  const { sequences, scores, logProbs, samplingProbs, residues, timingMs } =
    await engine.design(structureText, {
      temperature: 0.3, numSequences: 2, seed: 42,
      designChains: ['C'], fixedPositions: ['C8', 'C9'],
      omitAAs: 'C', useAtomContext: true, useSideChainContext: true,
    });
} finally {
  engine.destroy();
}
```

`shared/` and `webgpu/` contain no Node built-ins. The browser module Worker in
`demo/worker.js` is tested on real Chromium. Sequences are concatenated in parsed
protein-residue order; `residues` maps each character to its author chain, number
and insertion code. Scores have `{global, ligand}` fields. Log-probability arrays
are flattened `[L,21]` in `ACDEFGHIKLMNPQRSTVWYX` order; sampling probabilities are
`[L,20]` with X omitted. `timingMs` includes parsing, preprocessing and inference,
excluding initial model download. Each sequence uses seed + sequence index.
A supplied signal can cancel loading and is checked between decoding groups.
Concurrent calls on one engine are rejected; use separate engines for concurrency.

Preprocessing in JavaScript parses structures, computes virtual CB coordinates,
selects protein neighbours and closest context atoms, and constructs constraint
masks. RBFs, local-frame angle features, periodic element encodings, learned
projections, every protein/context encoder, and every decoder run on WebGPU.
Only the 21 decoder logits are read back per residue for categorical sampling;
there is no CPU neural-network fallback in the GPU path. Traces are opt-in.

Compute uses tiled 16x16 matrix multiplication with FP32 accumulation. FP16
weights use packed u32 storage and either core `unpack2x16float` or optional
`shader-f16` bitcasts. Activations and normalization stay FP32. INT8 is dequantized
once to FP32 on load. Packed-half models currently retain their original FP32
weight buffers as well; the weight-file size reduction is a transfer benefit,
not a claim of proportional GPU-memory reduction. Activation scopes retire
intermediate buffers after submission; `destroy()` releases all owned buffers and
an internally created device, while preserving an externally supplied device.

The loader verifies shard lengths and SHA256 (Web Crypto in secure contexts), and
uses the Cache API keyed by shard checksum. Cache denial/quota failure falls back
to fetching. OPFS is not needed and is not implemented. Remote servers must permit
CORS; use HTTPS or localhost for WebGPU and Worker support.

## Validation and demo

```sh
export PATH="/home/ruh/.nvm/versions/node/v22.21.1/bin:$PATH"
npm test          # CPU + parsers + bundle integrity + real Dawn hardware
npm run test:cpu
npm run test:gpu   # fails if hardware/weights/browser are unavailable; no skips
python3 -m http.server 8787 --bind 127.0.0.1
# Open http://127.0.0.1:8787/demo/
```

`npm test`: **19/19 passed**, zero skips. CPU suite: 11/11; GPU suite: 2/2;
SHA256/dtype/shard-cap checks: 6/6. Tested on Linux ARM64, NVIDIA GB10, Chromium
1243, including real `shader-f16`. Set `CHROME_PATH` to use a different binary.
The runner uses ephemeral HTTP ports and an isolated temporary browser profile;
it never uses ports 5173 or 5174. This host required `--use-angle=vulkan` and
`--use-vulkan=native` in addition to the Genie3 flags, including
`--enable-dawn-features=vulkan_enable_f16_on_nvidia`. Software adapters fail the
hardware test rather than silently supplying benchmark numbers.

The second real holo fixture is [3HS4](https://www.rcsb.org/structure/3HS4), human
carbonic anhydrase II with acetazolamide and zinc. Its 32 nearest zinc-site
residues retain the original small molecules and metal coordinates. Both CPU and
GPU match its PyTorch logits and exactly replay `SPLSVDTSHTHSPCHSETHIVVIYTGLTVNCA`.
RNA heavy atoms use the same context pathway as DNA; no independent RNA crystal
structure oracle is included.

### Recorded hardware results

Measured on the NVIDIA GB10; full data is in `fixtures/webgpu_metrics.json`.

| Check | relRMS | Maximum absolute error |
|---|---:|---:|
| FP32 teacher-forced logits vs PyTorch | 1.391641204e-06 | 1.096725464e-05 |
| Worst FP32 autoregressive trace | 3.483467332e-06 | 9.536743164e-06 |
| FP16 logits vs rounded PyTorch (both GPU paths) | 1.372527693e-06 | 1.358985901e-05 |
| FP16 logits vs original FP32 | 5.950729225e-04 | 4.675865173e-03 |
| Small-molecule + zinc FP32 logits | 1.584011207e-06 | 9.402632713e-06 |

Both half-weight paths exactly replay the ordinary FP32 oracle sequence with the
saved CDF draws. All four FP32 constraint/context sequences replay exactly. The
020 FP16/INT8 and 010 INT8 encoders/decoders are additionally checked against
precision-matched PyTorch traces. 605 generated shader compilations across
runtime instances and six dedicated shader checks pass with zero Dawn validation
errors (the count includes repeated source in separate instances).

Warm L=32 FP32 inference, excluding trace downloads and model loading: three runs
of 256.0, 263.3, 262.9 ms, mean **260.73 ms/sequence**.
Tracked peak GPU buffer allocation during the traced suite was
333,870,228 bytes (333.87 MB).
The full, uncropped 93-residue `inputs/1BC8.pdb` also completes through the public
API: 1764.2 ms on its first call (new shape pipelines),
**632.9 ms** warm, with identical seeded sequences.
This full-length check is a smoke/latency test; strict per-layer oracle comparisons
use the 32-residue fixtures. Timing varies with browser scheduling and other load;
the numbers above are wall-clock end-to-end measurements, not GPU timestamp queries.

### Numerical tolerance exception

FP32 runtime traces pass relRMS <1e-4. Both FP16 implementations also pass that
runtime tolerance against identically rounded PyTorch weights, including all
protein encoders and teacher-forced decoders. However, the requested blanket
FP16-versus-FP32 relRMS <5e-4 is **not met** by the 010 bundle: upstream PyTorch
itself produces 6.808270e-4 at encoder layer 0 h_V and 5.953310e-4 at final logits
with these rounded weights. The test records `requestedLogitsToleranceMet: false`
explicitly; its runtime-parity pass must not be interpreted as passing the tighter
weight-conversion criterion. INT8 quality differences are measured separately in
the export table, not hidden in a relaxed runtime tolerance.

## Vendoring and weight hosting

Copy `webgpu/` and `shared/` together, preserving their relative paths, with the
consumer's `sync-upstream.mjs`. Copy `cpu/` too if the reference/fallback is wanted.
No build step, runtime package dependency, Python, or network inference service is
required. Include this repository's MIT `LICENSE`. This implementation did not
modify ProteinDesignSpace or push any remote.

Publish the **contents** of the local `weights/` directory into a Hugging Face
model repository, preserving `<variant>/<precision>/manifest.json` and shard
paths. Set `baseUrl` to `https://huggingface.co/<owner>/<repo>/resolve/<revision>`
(or append `/weights` if that directory is retained on the host). Prefer an
immutable revision. No Hugging Face repository was provided, so bundles remain
local and no upload was performed.

Each variant's complete weight bundle, including manifest, is 10,524,212 bytes
(FP32), 5,325,177 bytes (FP16), or 2,800,942 bytes (INT8). The shared + WebGPU source
modules total 53,630 bytes uncompressed; the optional CPU backend adds 3,920 bytes.
`npm pack --pack-destination .cache` produces a self-contained runtime package
excluding checkpoints, fixtures, Python dependencies, and caches.

Sidechain packing (`ligandmpnn_sc`) remains outside this port. Fixed sidechain
context is implemented and parity-tested; it does not generate sidechain torsions
or packed all-atom coordinates.
