# AntiFold + ESM-IF1 WebGPU

One dependency-free ES module engine runs the complete **141,662,150-parameter
GVP-Transformer** with either AntiFold or base ESM-IF1 weights. All geometry,
neighbor selection, GVP message passing, transformer operations, probability
calculation, and sampling distributions execute in WGSL. No remote inference,
ONNX, WASM, or runtime npm dependencies. The CPU reference is a separate,
portable JavaScript implementation of the same execution graph.

Reference: [oxpig/AntiFold](https://github.com/oxpig/AntiFold/tree/9918344067f34a6b026f2634fdba212fc3e6e986),
commit **`9918344067f34a6b026f2634fdba212fc3e6e986`**. The upstream Python sources
are unchanged. The repository's BSD-3-Clause license and the MIT notices for
Meta ESM and GVP code are preserved in `LICENSE`.

## Architecture and supported inputs

See [ARCHITECTURE.md](ARCHITECTURE.md) for the source audit and exhaustive mode
matrix. All four GVP blocks, eight transformer encoder blocks, and eight
transformer decoder blocks are retained. GVP nodes have 1024 scalar and 256
vector channels; edges have 32 scalar and one vector channel. Graphs use 30
nearest neighbors. Transformers use width 512, eight heads, and ReLU FFNs of
width 2048. Outputs have the exact 35-token ESM-IF1 vocabulary.

The encoder combines masked-token, confidence-RBF, dihedral, local-frame GVP,
and local-frame raw geometry embeddings. Dihedral normalization uses unbiased
variance; other scalar normalization is LayerNorm. GVP vector normalization
retains upstream's hard nonzero-vector mask. Inference disables dropout.

PDB and mmCIF model 1, multi-character mmCIF chains, signed residue numbers,
insertion codes, first alternate conformers, MSE/SEC/UNK, missing atoms,
confidence inputs, and explicit ordered protein context chains are supported.
The original residue IDs survive all outputs. Numbering gaps do not invent
residues; chain concatenation inserts ten NaN-coordinate `-` positions. The
legacy terminal coordinates are infinity. Ligands, waters, ions, nucleic acids,
hydrogens, and sidechains do not enter this backbone-only model.

**Numbering is supplied by the caller.** ANARCII is a separate optional model
invoked by the upstream CLI, not part of the GVP-Transformer. This package
implements the upstream `--skip_anarcii_numbering` contract and accepts explicit
IMGT or Kabat annotations. It does not automatically detect antibodies or
convert numbering schemes. Kabat region boundaries and exact insertion-code
selectors are documented port extensions. No inference layer is pruned.

## Reproduce weights and oracles

```bash
mkdir -p .cache/uv .cache/python .cache/tmp .cache/bin
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
export UV_PYTHON_BIN_DIR="$PWD/.cache/bin"
export TMPDIR="$PWD/.cache/tmp"
uv python install 3.12.12
uv venv .venv --python 3.12.12 --managed-python
uv pip install --python .venv/bin/python -r tools/requirements-export.lock
.venv/bin/python tools/download_weights.py
.venv/bin/python tools/export_models.py
.venv/bin/python tools/export_residue_names.py
.venv/bin/python tools/make_cases.py
.venv/bin/python tools/parser_oracle.py
.venv/bin/python tools/dump_oracle.py
```

PyTorch 2.9.1, NumPy 1.26.4, Biotite 1.5.0, torch-geometric 2.7.0; the complete
resolved environment is pinned in `tools/requirements-export.lock`. Official
checkpoint downloads are pinned to full SHA-256 digests in the downloader.
AntiFold's digest is also published in its upstream source; ESM-IF1's digest
pins the official Meta download observed during this port. Export runs on CPU
with eight threads and strict state-dict loading.

Each variant has **489 state tensors**, including two nonnumerical positional
embedding dtype markers exported as zero. They are not learned parameters.
FP32 reload is bit-identical to original-checkpoint inference in every case.
All precisions reload into the original PyTorch model before oracle execution.

<!-- EXPORT_TABLE -->
| Variant | Precision | Shards | Bytes | Weight relRMS |
| --- | --- | ---: | ---: | ---: |
| antifold | fp32 | 31 | 566,704,576 | 0 |
| antifold | fp16 | 14 | 283,837,372 | 0.000206429 |
| antifold | int8 | 6 | 143,013,706 | 0.0125827 |
| esmif1 | fp32 | 31 | 566,704,576 | 0 |
| esmif1 | fp16 | 14 | 283,837,372 | 0.000206381 |
| esmif1 | int8 | 6 | 143,209,146 | 0.0125903 |
<!-- /EXPORT_TABLE -->

The `gvp-transformer-safetensors-v1` manifest records variant, precision, upstream
commit, parameter/tensor counts, configuration, vocabulary, exact bytes and
SHA-256 per shard, and row-part mappings. Every serialized shard, **including
its header**, is at most **25,000,000 bytes**. Multipart tensors reconstruct
contiguous rows without transposition. Every I8 part has F32 per-output-row
scales in the **same shard**. Biases, normalization vectors, embeddings, and
geometry buffers remain F32. F16 and packed 4×I8 storage use F32 arithmetic.
The portable F16 path uses `unpack2x16float`; `useF16: true` selects native
`shader-f16` storage loads when supported.

ESM-IF1 INT8 retains one 256×256 matrix in F32:
`encoder.gvp_encoder.encoder_layers.0.conv.message_func.2.wv.weight`.
With that matrix quantized, a vector's squared norm plus epsilon was
`0.00020000231`, only about `2.3e-9` above the hard `2e-4` mask threshold.
Equivalent CPU/WGSL GEMM reductions crossed that threshold, causing GVP trace
relRMS `4.59e-4`. Retaining this projection changes the bundle by about 196 KB
and restores parity. Neither the architecture nor the tolerance is changed.
Other matrices use the requested precision.

<!-- QUALITY_TABLE -->
| Variant | Precision | Worst logit relRMS vs original FP32 (all modes) |
| --- | --- | ---: |
| antifold | fp32 | 0 |
| antifold | fp16 | 0.000248228 |
| antifold | int8 | 0.0194742 |
| esmif1 | fp32 | 0 |
| esmif1 | fp16 | 0.000433103 |
| esmif1 | int8 | 0.050726 |
<!-- /QUALITY_TABLE -->

Quantization error is measured against original FP32; hardware arithmetic
parity is measured **separately against identically quantized PyTorch**. INT8
can noticeably change preferences on difficult/missing-coordinate inputs.
There is no claim of experimental sequence quality or FP32-equivalent ranking.

## API

```js
import { createGVPTransformerEngine } from './webgpu/index.js';
const controller = new AbortController();
const engine = await createGVPTransformerEngine({
  baseUrl: '/weights',         // contains antifold/ and esmif1/
  variant: 'antifold',        // or 'esmif1'
  precision: 'fp16',          // fp32 (default), fp16, int8
  cache: 'cache',             // none (default), cache, opfs
  useF16: false,              // optional native f16 storage reads
  signal: controller.signal,
  onProgress: event => console.log(event.phase, event.loaded, event.total),
});
try {
  const result = await engine.predict(pdbText, {
    chains: ['H', 'L'], heavyChain: 'H', lightChain: 'L', numbering: 'imgt',
    signal: controller.signal, returnIntermediates: false,
  });
  console.log(result.logLikelihood, result.perplexity, result.logits);
} finally {
  engine.destroy();
}
```

`createAntiFoldEngine` is an alias; `createESMIF1Engine` selects base ESM-IF1.
The CPU equivalents are exported from `cpu/index.js`. Raw inputs may be
`{coords: Float32Array(L*9), sequence, residues, confidence}` or the result of
`parseStructure()` / `combineChains()`. Null coordinates in JSON are missing
values. Provide `targetChain` for target-first complex concatenation and
chain-specific score aggregation. Input admission is 1–2048 residues and also
depends on the device's actual buffer limits.

`predict()` / `score()` returns F32 logits `[L,35]`, token log probabilities,
canonical probabilities `[L,20]`, per-position entropy perplexities,
all-target and finite-coordinate mean log likelihood/perplexity, residue IDs,
and `timingMs`. Gap/context residues are excluded from aggregate scores.
`sequence` overrides the teacher-forced sequence to score single, double, or
arbitrary mutants. `returnIntermediates` adds every major GVP/transformer trace.
All result arrays can be transferred to another worker with their `.buffer`.

Two distinct sampling APIs preserve the upstream behaviors:

- `sample(input, {uniforms, temperature, fixed, length, regions, ...})` performs
  left-to-right autoregressive decoding. `fixed` maps zero-based concatenated
  positions to vocabulary tokens or token IDs. Explicit `regions` fixes all
  positions outside the selected regions to the native sequence. Target-chain
  sampling defaults to the target's length. Supply exactly one uniform `[0,1)`
  per sampled position, including fixed positions. Encoder output is reused
  throughout decoding; decoder prefixes are recomputed for correctness (no KV
  cache optimization). `returnIntermediates` returns the logits at each draw.
- `sampleIndependent(input, {uniforms, temperature, regions, fixed,
  limitExpectedVariation, backmutationOrder, ...})` samples canonical residues
  from native-sequence-conditioned AntiFold logits. This matches upstream CDR
  sampling. Uniforms cover all concatenated positions. When limiting variation,
  provide an explicit permutation of residue indices for backmutation replay;
  the default order is ascending. Framework/context positions remain fixed.

Regions include `CDR1`, `CDRH3`, `CDRL3`, `CDRH`, `CDRL`, `FW1`–`FW4`, `FWH`,
`FWL`, `all`, `allH`, `allL`, chain-prefixed integer ranges, and exact insertions
such as `H:109A`. Integer selections include insertion variants. A heavy-chain
ID is required for antibody region selection. Kabat CDRs are H31–35/H50–65/
H95–102 and L24–34/L50–56/L89–97; IMGT CDRs are 27–38/56–65/105–117.

A caller-supplied `device` remains alive on teardown. Self-created devices are
destroyed. `destroy()` is idempotent. `memoryUsage()` reports total and transient
owned buffers/bytes; transient allocations return to zero after every call,
including cancellation and errors. Scratch buffers are pooled during inference
and destroyed in `finally`. Buffer admission happens before inference dispatch.
Already-submitted GPU work is awaited before cancellation frees its resources.
Concurrent calls on one engine reject. Use immutable versioned base URLs when
publishing revised bundles so persistent cache entries remain unambiguous. The CPU reference caches a matching
encoder result across scoring/sampling calls and invalidates it on coordinate
or confidence changes.

## Validation on real hardware

```bash
npm test
npm run test:cpu
npm run test:gpu
```

No installation is needed for JavaScript tests. Missing weights, fixtures,
Chromium, hardware WebGPU, or numerical parity **fail**, never skip. The harness
binds an ephemeral loopback port with `listen(0)`. Chromium profiles, caches,
and temporary files stay inside this repository. Set `CHROME_PATH` to override
`~/.cache/ms-playwright/chromium-1243/chrome-linux-arm64/chrome`.

Test hardware: **ARM64 NVIDIA GB10**, driver **580.159.03**, Vulkan WebGPU.
Chromium reports vendor `nvidia`; the device name is verified with `nvidia-smi`.
Software/fallback adapters are rejected. Working flags:

```text
--headless=new --no-sandbox --disable-gpu-sandbox --ozone-platform=headless
--enable-unsafe-webgpu --ignore-gpu-blocklist --enable-features=Vulkan
--use-angle=vulkan --use-vulkan=native --disable-vulkan-surface
--enable-dawn-features=vulkan_enable_f16_on_nvidia
```

The child browser unsets DISPLAY/WAYLAND_DISPLAY and uses the NVIDIA Vulkan ICD
when present. The short project-local `.t/` temporary directory avoids Linux
Unix-domain socket pathname limits.

<!-- MODE_TABLE -->
| Case | Model | Residues incl. linkers | CPU FP32 worst trace relRMS | GPU worst trace relRMS (all precisions) | Replay |
| --- | --- | ---: | ---: | ---: | --- |
| antibody_hl | antifold | 21 | 3.06e-06 | 9.68e-06 | Exact AR + CDR |
| nanobody | antifold | 4 | 4.47e-06 | 5.09e-06 | Exact AR + CDR |
| monomer | esmif1 | 6 | 1.24e-05 | 3.4e-05 | Exact AR |
| score_single | esmif1 | 6 | 1.41e-05 | 3.33e-05 | Exact AR |
| score_double | esmif1 | 6 | 1.56e-05 | 3.33e-05 | Exact AR |
| complex | esmif1 | 21 | 7.66e-06 | 2.02e-05 | Exact AR |
| antibody_antigen | antifold | 19 | 2.25e-06 | 7.09e-06 | Exact AR + CDR |
| small_full | esmif1 | 20 | 1.18e-05 | 9.45e-05 | Exact AR |
| antibody_hl_full | antifold | 236 | 3.27e-06 | 7.65e-06 | Exact AR + CDR |
| vhh_full | antifold | 105 | 4.43e-06 | 7.34e-06 | Exact AR + CDR |
| missing_insertion | esmif1 | 21 | 1.81e-05 | 9.03e-06 | Exact AR |
| antibody_kabat | antifold | 21 | 3.06e-06 | 9.68e-06 | Exact AR |
| parser_pdb | esmif1 | 16 | 7.69e-06 | 1.89e-05 | Exact AR |
| parser_cif | esmif1 | 16 | 7.69e-06 | 1.89e-05 | Exact AR |
<!-- /MODE_TABLE -->

Each mode checks preprocessing, graph membership, both GVP streams, every
encoder/decoder block, logits, probabilities, entropy, and exact autoregressive
token replay at temperature 0.7 with fixed positions. The reference uses the
original incremental PyTorch decoder; the port recomputes causal prefixes.
Native-conditioned IMGT CDR sampling and backmutation replay call the original
AntiFold functions with explicit uniforms/permutations injected at the random
sampling boundary. Seeds are never assumed equivalent across languages.
Kabat selection is tested independently from upstream IMGT selection.

Full machine-readable evidence is in `fixtures/{export,cpu,gpu,browser}_metrics.json`.
Independent parser oracles cover altLoc, MSE/SEC/UNK, model selection, negative
IDs, insertion codes, missing atoms, and ignored molecular context. Browser tests
execute real module workers, transfer typed outputs, prime then block network
for Cache API/OPFS hits, and inject both SecurityError and QuotaExceededError.
They verify load/inference/submission cancellation, thrown-error cleanup,
concurrency rejection, external device ownership, native f16 loads, and buffer
admission. The static demo also runs a real model in Chromium.

<!-- TIMING_TABLE -->
| Model / input | Precision | Cold weight load (ms) | Warm median (ms) |
| --- | --- | ---: | ---: |
| esmif1 / small_full (20) | fp32 | 2119.9 | 156.0 |
| esmif1 / small_full (20) | fp16 | 1106.6 | 161.7 |
| esmif1 / small_full (20) | int8 | 617.6 | 146.9 |
| antifold / antibody_hl_full (236) | fp32 | 2189.5 | 656.8 |
| antifold / antibody_hl_full (236) | fp16 | 1057.3 | 763.3 |
| antifold / antibody_hl_full (236) | int8 | 502.5 | 731.6 |
| antifold / vhh_full (105) | fp32 | 2189.5 | 473.1 |
| antifold / vhh_full (105) | fp16 | 1057.3 | 395.3 |
| antifold / vhh_full (105) | int8 | 502.5 | 326.3 |
<!-- /TIMING_TABLE -->

Warm measurements are three complete predictions after initial compilation,
including command preparation and readback. They exclude loading and are not
exclusive-GPU benchmarks. Cold loads and individual samples are retained in
`fixtures/gpu_metrics.json`.

## Demo, packaging, and integration

Serve the project from any available local HTTP port and open `demo/index.html`.
The demo accepts PDB/mmCIF files, both models and all precisions, score/design
modes, regions, numbering, temperature, fixed residues, cancellation, a residue
preference plot, FASTA/CSV/input-structure downloads, and explicit GPU release.
Weights are downloaded from the same origin; structures remain local.

```bash
node sync-upstream.mjs .cache/vendor-smoke
node sync-upstream.mjs .cache/vendor-smoke --write
```

The vendor allowlist copies only `shared/`, `webgpu/`, `cpu/`, `LICENSE`, and
this engineering specification. It excludes weights, fixtures, Python, caches,
and environments. The write path smoke-imports the copied runtime. The npm
package uses the same runtime allowlist.

Proposed `denovo.run` artifact mapping:

| Input / output | Port representation |
| --- | --- |
| StructureArtifact | PDB/mmCIF text or N/CA/C coordinates plus stable chain/residue/insertion IDs |
| Collection of structures/chains | Explicit chain order, target chain, heavy/light roles and coordinate masks |
| SequenceSet input | Candidate sequences scored independently with the same structural context |
| SequenceSet output | Sampled sequences, model variant/precision, temperature, fixed positions and recorded replay uniforms |
| ResidueTrack output | Canonical probabilities, native/candidate log probabilities, entropy perplexity and mutation/region selection |
| Collection output | Structure, sequences, per-residue tracks, aggregate score metadata and timings |

No integration destination is modified by this repository's tests.
