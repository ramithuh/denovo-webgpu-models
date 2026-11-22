# LASErMPNN browser port

This port targets the MIT-licensed [polizzilab/LASErMPNN](https://github.com/polizzilab/LASErMPNN) checkpoint `laser_weights_0p1A_nothing_heldout.pt`, upstream revision `e70f2c6d765416f7e29d51bfd6d4e08496438878`. The upstream Python sources remain unchanged. The FP32 CPU and WebGPU implementations are validated against upstream on the full protonated holo example. FP16 arithmetic also matches the identically quantized upstream model, but FP16 weight rounding exceeds the requested original-FP32 logit threshold. Hosting is prepared locally; no weights have been published.

## Architecture audit

The checkpoint contains **20,599,412 learned parameters**, **492 state tensors**, and its fine-tuned SPICE ligand encoder. The separate pretrained ligand checkpoint is for pretraining; inference uses the embedded encoder. Optimizer state is excluded.

- Ligand nodes: concatenated periodic-table period/group one-hot encoding (26 features); explicit input hydrogens are retained. Graphs use spatial neighbors, not inferred bond orders. Each ligand residue forms its own ligand graph; `k=5`, including self edges. Ligand vectors point toward the mean neighbor coordinate.
- Ligand encoder: 256 scalar channels, 15 vector channels, 128 edge channels, three HomoGATv2 layers with three attention heads. Distances use 75 RBFs from 0 to 15 Å plus `1e-4`. A three-layer gated GVP adapter maps ligand embeddings into the protein graph.
- Protein frames: idealized N/CA/C, virtual CB, and psi-dependent O in **N, CA, CB, C, O** order. Protein graph: 48 nearest CA neighbors including self; ligand→protein graph: up to 48 ligand atoms for any CA within 20 Å of a ligand atom. Protein edges use 25 pairwise atom distances × 16 RBFs (2–22 Å), plus 100 learned-vector dot products. Ligand–protein edges use five distances × 75 RBFs plus ten dot products.
- Protein encoder: three heterogeneous GAT layers, 256 scalar/10 vector channels, 128 edge channels, one attention head with 4× attention expansion, dense GELU message and residual updates. The heterogeneous sub-GATs have edge updating disabled; each encoder's final angular projection updates the edges.
- Autoregressive decoder: three heterogeneous layers. Previously decoded neighbors contribute their layer-specific hidden state, amino-acid embedding, and four circular χ embeddings. Future/self neighbors contribute the fixed encoder state, NotDecoded label, and zero χ embeddings. Sequence vocabulary is **ARNDCEQGHILKMFPSTWYVX**, plus embedding-only NotDecoded.
- Coupled χ decoding: four sequential 72-bin (5°) heads, a continuous offset regressor for each sampled bin, and gated vector updates between heads. Circular normalized Gaussian embeddings feed subsequent residues. Masks include rotatable polar hydrogens (e.g. tyrosine χ3).
- Reconstruction: upstream ideal coordinate/bond tables, NeRF extension with `1e-6` normalization epsilon, Kabsch alignment of leftover groups, delayed Tyr HH placement, residue-to-backbone alignment, and triad alignment of nonrotatable hydrogens. Backbone NH is a separate phi-dependent extension. Missing atoms/undefined χ angles use NaN padding.
- Fixed residues use `chain_mask=1`; sampled residues use zero. Upstream comments disagree in places; executable behavior is the reference. First-shell contact masks use sidechain–ligand heavy-atom distance <5 Å (glycine CA <5.3 Å). There is no native `firstShellOnly` argument; the browser control can derive a fixed mask from this definition.

## Export and oracle

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cpu
# See tools/requirements-oracle.txt for the complete tested environment.
# ARM64 requires building torch-scatter and torch-cluster against installed torch:
uv pip install --python .venv/bin/python setuptools wheel numpy safetensors prody pandas scikit-learn tqdm scipy h5py rdkit logomaker
MAX_JOBS=2 uv pip install --python .venv/bin/python --no-build-isolation torch-scatter torch-cluster
.venv/bin/python tools/export_lasermpnn.py
.venv/bin/python tools/dump_oracle.py
```

Exporter copies the official checkpoint already tracked in `model_weights/` into ignored `.cache/lasermpnn/`. `weights/`, `.venv/`, and `.cache/` are ignored. No checkpoint download is required for this checkout.

The `lasermpnn-safetensors-v1` manifest follows the Genie3 field structure: source/checksum, config, total parameters, tensor count, total shard bytes, shard checksums, and tensor→shard/dtype/shape mapping. Learned 2D matrices use F16 in the FP16 bundle; 1D norms/biases and geometry buffers remain F32. Integer/bool buffers use I32. INT8 learned matrices use symmetric per-row scales, stored as `<name>.__scale` F32 in the same shard. Higher-rank attention parameters remain F32. The **25,000,000-byte** shard limit includes the Safetensors header.

| Bundle | Shards | Bytes | Learned-weight relRMS | Full-state relRMS |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 4 | 82,544,532 | 0 | 0 |
| FP16 | 2 | 41,460,126 | 2.0599719767e-4 | 8.2553284442e-5 |
| INT8 | 1 | 21,119,471 | 1.0119734216e-2 | 4.0554789415e-3 |

`fixtures/oracle_lasermpnn.json` + `.bin` contain **81 tensors** from CPU FP32 inference on the complete repository example: **115 residues and 31 ligand atoms**. Binary records are little-endian F32/I32 with explicit shapes, byte offsets, and byte lengths. They include input frames/χ/metadata, all three graphs and distance features, ligand embeddings, each encoder layer's outputs, raw AA and all χ/offset logits in decoding order, final predictions, protonated coordinates, and backbone NH coordinates.

For portable stochastic replay, only `Categorical.sample` is temporarily replaced by inverse-CDF sampling using saved uniform draws. Network functions, preprocessing, decoding loop, and geometry remain upstream code. The fixture saves the decoding permutation, pre-sampling PyTorch RNG state, uniform draws and chosen category indices. This does **not** claim to reproduce PyTorch's native multinomial algorithm from a JS seed.

Learned-weight errors exclude geometry/index buffers; full-state errors include every checkpoint tensor. NaN masks are verified separately.

The forward quantization comparison fixes categorical choices to the FP32 trajectory, recomputing continuous offsets and all hidden states. Exact machine-readable results: `fixtures/export_metrics.json`.

| Output relRMS | FP16 | INT8 |
| --- | ---: | ---: |
| Ligand encoder scalars | 1.0538196030e-4 | 4.9918049495e-3 |
| Final protein encoder scalars | 3.5564658066e-4 | 1.7013772628e-2 |
| AA logits | 6.0460492735e-4 | 3.4571441965e-2 |
| χ1 logits | 5.2682897807e-4 | 3.1618471091e-2 |
| χ2 logits | 5.1422492840e-4 | 3.5556069665e-2 |
| χ3 logits | 1.5698464954e-3 | 6.1815549392e-2 |
| χ4 logits | 1.4297144820e-3 | 9.3506112369e-2 |
| χ angles, raw degrees | 2.7559654825e-6 | 3.9473419945e-1 |
| Protonated coordinates | 2.3925607010e-7 | 3.0779940876e-5 |

FP32 reloaded weights and all measured forward outputs have **zero error**. Raw angle error includes the discontinuity at ±180°; coordinate error is the more useful metric there. The requested FP16-vs-original-FP32 logit limit of 5e-4 is exceeded by upstream weight rounding itself. GPU arithmetic parity and weight-quantization error must be reported separately; the original threshold has not been silently relaxed.

## Validation status

Milestone 1: exported and strictly reloaded all three bundles; generated the full holo oracle and measured quantization error. Milestone 2 CPU parity and Milestone 3 NVIDIA/Dawn parity are implemented; see the measured scope and remaining FP16 quality limitation below. Hardware detected: NVIDIA GB10 on ARM64; installed Chromium is `~/.cache/ms-playwright/chromium-1243/chrome-linux-arm64/chrome`.

## CPU reference and structure IO

`shared/lasermpnn/` and `cpu/lasermpnn/` have no package dependencies. Shared code uses typed arrays, browser `fetch`, Web Crypto (when available), and optional Cache API. It contains no Node imports. Node applications inject a `read(name)` callback into `loadWeights` for local files.

```js
import { createLaserMpnnCpuEngine } from './cpu/lasermpnn/index.js';
const engine = await createLaserMpnnCpuEngine({ baseUrl: '/weights/fp32' });
try {
  const result = await engine.design(pdbOrMmcifText, {
    temperature: 0.1,
    chiTemperature: 0.1,
    numSequences: 1,
    seed: 42,
    fixedPositions: ['A:12'], // chain:resnum plus insertion code, if present
    designChains: ['A'],
    omitAAs: 'C',
    firstShellOnly: false,
  });
  // result: { sequences, scores, chiAngles, pdbTexts, timingMs }
} finally {
  engine.destroy();
}
```

Temperature zero selects argmax. Fixed positions preserve residue identity and available input χ angles; missing χ angles are sampled as upstream does. `firstShellOnly` derives a fixed mask using upstream's contact definition. `numSequences` uses successive uint32 seeds. Randomness uses a portable seeded generator, not the PyTorch RNG; oracle replay explicitly supplies saved decoding order and draws.

PDB and mmCIF parsers select the first model and blank/A alternate locations, preserve author residue identifiers/insertion codes, exclude water by default (`useWater: true` includes it), and retain ligand elements including hydrogens/metals. mmCIF uses `_atom_site` author fields with label-field fallbacks, quoted values, and semicolon-delimited multiline values. No protonation or chemical bond inference is performed. Supply the same protonated ligand expected by upstream. For PDB output, long mmCIF chain identifiers and duplicate chain names across segments are mapped to distinct one-character identifiers, with `REMARK 950` records preserving the original segment/chain mapping. The PDB format limit is 62 assigned nonblank chain IDs; residue numbers/insertion codes must fit PDB fields. Output includes protonated sidechains, phi-dependent backbone NH, and the original ligand coordinates. The CLI's final hydrogen cleanup is also applied: Cys HG is removed when SG atoms lie within 2.5 Å, and His HD1 is removed to select the epsilon tautomer, matching executable upstream behavior.

Scores are natural-log probabilities from untempered model logits, with omitted identities masked: sequence, χ-bin, total, and per-residue values. χ scores describe the discrete sampled bin; the continuous offset has no probability density head. Fixed χ scores use the nearest input bin. They are not binding-energy predictions.

`npm run test:cpu`: **4 tests passed**, zero skipped (full oracle trace, PDB/feature parity, mmCIF equivalence, and API/control checks). Full 115-residue CPU trace took 41.7 seconds on the test host.

| CPU FP32 vs upstream | relRMS |
| --- | ---: |
| Idealized backbone | 7.9578471566e-8 |
| Protein edge distances | 2.6721548239e-7 |
| Ligand–protein distances | 1.5814107375e-7 |
| Final protein encoder scalars | 5.1529090523e-7 |
| AA logits | 9.5154172422e-7 |
| Worst χ head | 1.4172399831e-6 |
| χ angles | 4.8903946413e-8 |
| Protonated sidechain coordinates | 1.1635626246e-8 |

All graph indices and sampled sequence indices match exactly. Tests check NaN masks explicitly and enforce relRMS <1e-4, including all intermediate ligand/protein encoder traces. Typed-array operator outputs are FP32; the JS reference accumulates dot products in JS numbers before rounding to FP32. Geometry uses JS double arithmetic, matching upstream's double-precision rotamer construction before fixture serialization to F32.

## WebGPU engine

```js
import { createLaserMpnnEngine } from './webgpu/index.js';
const abortController = new AbortController();
const engine = await createLaserMpnnEngine({
  baseUrl: '/weights/fp32', // URL of the precision directory containing manifest.json
  variant: 'laser-0p1A-nothing-heldout',
  precision: 'fp32',       // fp32, fp16, or int8
  onProgress: (event) => console.log(event),
  signal: abortController.signal,
});
try {
  const result = await engine.design(structureText, { seed: 42 });
} finally {
  engine.destroy();
}
```

All neural operators execute in WGSL: tiled GEMM, vector transforms/norms/gates, erf-based GELU, layer normalization, RBF encoding, angular features, attention softmax/reduction, gathers, and edge/residual updates. Parsing, spatial graph construction, stochastic sampling, and double-precision rotamer geometry execute in shared JavaScript. Autoregressive logits are read back for sampling; intermediate neural activations remain on the GPU. The public API does not invoke Python, a server inference endpoint, WASM, ONNX, or a CPU neural fallback.

The default uses core WebGPU F32 arithmetic. FP16 bundles work on a device without `shader-f16` by widening their stored weights. For optional half-precision weight storage in the GEMM shader, pass `precision: 'fp16', shaderF16: true`; accumulation remains F32. INT8 bundles are dequantized on load and use the F32 operator path. CPU-side decoded weights remain resident for lookup tables and sampling geometry.

Pass an existing `GPUDevice` to share an application device. `destroy()` frees all activation, weight, uniform and staging buffers; it destroys an internally created device but preserves a caller-owned device. Repeated calls to `destroy()` are harmless. One `design()` call runs at a time per engine. Abort signals are checked during loading and between autoregressive residues; cleanup runs on errors/cancellation. The Cache API is used when available, with SHA256 shard verification; use `cache: false` to disable it. No OPFS permission is needed.

### Hardware validation and performance

Tests run headless Chromium 1243 with Dawn on an **NVIDIA GB10 / ARM64**. Adapter vendor is `nvidia`, `isFallbackAdapter=false`; software fallback is rejected. The ARM64 host requires these flags in addition to the Genie3 reference flags:

```text
--headless=new --no-sandbox --disable-gpu-sandbox --ozone-platform=headless
--enable-unsafe-webgpu --ignore-gpu-blocklist --enable-features=Vulkan
--use-angle=vulkan --use-vulkan=native --disable-vulkan-surface
--enable-dawn-features=vulkan_enable_f16_on_nvidia
```

The harness binds an ephemeral loopback port, uses its own temporary Chromium profile, and closes its processes after each test. It does not use ports 5173 or 5174. Set `CHROME_PATH` to override browser discovery.

`test/webgpu_shaders.test.mjs` compiles **15 unique WGSL variants**, exercising a core device without requesting f16 and a separate device with `shader-f16`. `test/webgpu_parity.test.mjs` checks every FP32 ligand/protein encoder intermediate, coupled AA/χ logits, exact sequence replay, χ angles, protonated coordinates, public PDB-to-design inference, PDB output coordinates, sequence/χ log-likelihoods, teardown, and borrowed-device survival. FP16 core and shader-f16 are checked against `oracle_lasermpnn_fp16.*` at relRMS <5e-4. There are no skipped hardware tests when the required environment is missing: they fail with a diagnostic.

FP16 fixtures replay the FP32 categorical choices while recomputing all continuous states. This isolates arithmetic fidelity. **The original FP16-vs-FP32 5e-4 quality requirement is not met**: upstream rounding alone gives AA logits 6.046e-4 and worst χ-head logits 1.570e-3. The test report records this separately as `fp16OriginalFp32LimitMet: false`. FP16 is experimental; the test threshold is not evidence that quantization meets the original quality target.

Measured hardware metrics and wall-clock timings are saved by the tests to ignored `test-results/webgpu.json`. The committed verification snapshot and exact timings are in `fixtures/validation_results.json`. Latency includes upload-on-first-use, dispatch, and autoregressive readbacks; weight download, browser startup and pipeline creation are excluded. Public API timings also include PDB parsing, featurization, hydrogen reconstruction and serialization. These are measurements for one complete 115-residue / 31-ligand-atom example, not a scaling benchmark.

## Demo, vendoring and weight hosting

```bash
# Node 22 is required for the test harness; runtime modules are browser-native.
npm run test:cpu
npm run test:gpu
npm test

# Serve the repository, then open http://127.0.0.1:8080/demo/
python3 -m http.server 8080 --bind 127.0.0.1

# Create a NEW staging directory for an application's sync process:
node tools/sync-upstream.mjs --out .cache/vendor-staging/lasermpnn
```

The staging script copies `shared/`, `cpu/`, `webgpu/`, the package exports, documentation, and MIT license. It excludes weights, Python tooling, fixtures, test dependencies, and caches, and refuses to overwrite an existing destination. It does not modify ProteinDesignSpace. Runtime files have no npm dependencies; the full source payload is approximately 78 KB (22 KB gzip, measured by concatenation rather than a bundler).

The local `weights/` tree is ready for a Hugging Face model repository. A model card is tracked at `docs/HUGGINGFACE_MODEL_CARD.md` and copied to `weights/README.md`, with the upstream license at `weights/LICENSE`. Host the precision subdirectories unchanged, then use:

```text
https://huggingface.co/<owner>/<repo>/resolve/<pinned-revision>/fp32
```

No Hugging Face namespace or authenticated publishing destination was supplied, so no upload has been performed. No git remote has been pushed. The local bundle sizes and SHA256 manifests are the deliverables available for publication.

### Final verification snapshot

`npm test`: **7 passed, 0 failed, 0 skipped**; **15 shader variants**, zero Dawn validation errors.

| Measurement | Value |
| --- | ---: |
| GPU FP32 AA logits relRMS | 8.8720779127e-07 |
| GPU FP32 χ4 logits relRMS | 1.5325104931e-06 |
| GPU FP32 raw protonated coordinates relRMS | 1.1437486422e-08 |
| Public API sequence log-probability relRMS | 2.0980023860e-05 |
| Public API χ log-probability relRMS | 5.4760427958e-06 |
| Public API PDB coordinates relRMS | 9.7354171624e-06 |
| FP32 encoder + decoder, 115 residues | 5116.6 ms |
| fp16-core encoder + decoder | 4911.5 ms |
| fp16-shader encoder + decoder | 5575.3 ms |
| Full PDB → design API, three successive runs | 4953.7, 4849.9, 4888.3 ms |
| Runtime JS source / gzip concatenation | 77,864 / 21,837 bytes |
