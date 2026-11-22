# ThermoMPNN WebGPU

Dependency-free browser inference for the published ThermoMPNN point-mutation stability model. All neural operations and geometric features execute in WGSL. No remote inference, WASM ML, ONNX, or runtime npm dependencies. `cpu/` is an independent FP32 numerical reference. Negative predicted ΔΔG values indicate stabilization; outputs are in the upstream model's kcal/mol convention.

Upstream: [Kuhlman-Lab/ThermoMPNN](https://github.com/Kuhlman-Lab/ThermoMPNN/tree/370f76ec62bd929f7425e311d8df04a0d094990f), commit **`370f76ec62bd929f7425e311d8df04a0d094990f`**, MIT. The inherited Python source files are unchanged and their bytes are checked against that commit before oracle/export execution.

## Architecture and supported scope

The published default checkpoint has **4,342,876 parameters / 130 tensors**. The inference bundle has **1,835,740 parameters / 128 tensors**. The complete audit and mode inventory were written before the JS/WGSL implementation in `ARCHITECTURE.md` in the source checkout.

- Four backbone atoms (N, CA, C, O), virtual Cβ, 48 nearest Cα neighbors. Twenty-five distance pairs × 16 radial basis functions, plus 16 positional features; 128 hidden channels.
- Three node/edge encoder blocks, three node decoder blocks, erf-form GELU (a polynomial erf approximation accurate near FP32 precision), LayerNorm ε=1e-5, message sum /30. Coordinate noise and dropout are disabled.
- The pinned ThermoMPNN backbone explicitly replaces its causal decoder mask with **all-visible attention**. No sampling or stochastic replay is involved. Readout concatenates decoder layers **2, 1**, then the WT sequence embedding.
- LightAttention is called on **one residue at a time** by upstream. Its softmax is therefore one, and only the feature convolution's center slice can contribute. The unused attention convolution and eight inactive feature-kernel slices are omitted. Round trips into the unchanged full upstream model show exactly zero FP32 output error. All three decoder blocks and the optional diagnostic sequence logits are retained.
- Head: 384→384 feature projection, ReLU/linear 384→64→32→21, scalar affine calibration, then mutant minus WT subtraction. Calibration occurs before subtraction, preserving upstream FP32 rounding.

Full saturation returns an **L×20** row-major Float32Array in `ACDEFGHIKLMNPQRSTVWY` order. Targeted scoring selects arbitrary point mutations from the same context-dependent matrix. WT-to-WT entries are exactly zero. Double-mutant epistasis and indels require the separate ThermoMPNN-D/I models and are outside this checkpoint. No ligand/ion/nucleic-acid conditioning, sequence-only scoring, packing, or sequence generation is supported by ThermoMPNN's forward path.

PDB parsing matches upstream: sorted author residue numbers, lexical insertion codes, padded missing-number gaps, first atom occurrence regardless of altLoc/occupancy, HETATM MSE→MET, other HETATM ignored. Unknown ATOM residues become X; incomplete backbone atoms become zero coordinates and mask=0. `residues` preserves chain/author number/insertion code separately from model indices. `valid` excludes gaps/unknowns; `mask` reports complete backbone geometry. Raw matrix entries remain available at masked positions; ranking and CSV omit gaps/unknowns. A canonical site with missing backbone remains scoreable as upstream, and is explicitly flagged.

mmCIF is an input extension: atom_site author IDs (label-ID fallback) become the same model tensors, while preserving multi-character chain names. Paired PDB/mmCIF fixtures prove feature equality, including Alpha/Beta chain IDs. Chain selection filters structure **before** graph construction and preserves the requested order. Selecting a chain removes other chains' geometric context; omit `chains` to retain a complex. Multiple coordinate models use first atom occurrence semantics, matching the upstream parser; there is no ensemble averaging. Input admission is capped at 2,048 positions including gaps and then checked against device buffer limits.

## API

```js
import { createThermoMPNNEngine } from './webgpu/index.js';

const controller = new AbortController();
const engine = await createThermoMPNNEngine({
  baseUrl: '/weights/default', // fp32/, fp16/, int8/ underneath
  precision: 'fp32',         // fp16 or int8 also supported
  cache: 'cache',            // none | cache | opfs
  useF16: false,             // optional native shader-f16 storage loads
  signal: controller.signal,
  onProgress: ({ phase, loaded, total }) => console.log(phase, loaded, total),
});
try {
  const saturation = await engine.predict(pdbOrMmcifText, {
    mode: 'ssm',
    // chains: ['Alpha', 'Beta'],  // omit to retain every chain
    signal: controller.signal,
  });
  console.log(saturation.ddg, saturation.shape, saturation.residues);
  const selected = await engine.predict(pdbOrMmcifText, {
    mode: 'targeted',
    mutations: [
      { chain: 'A', number: 100, ins: 'A', wildtype: 'G', mutation: 'V' },
      { index: 0, mutation: 'A' }, // alternative: zero-based model index
    ],
  });
  console.log(selected.scores, selected.mutations, selected.timingMs);
} finally {
  engine.destroy();
}
```

The factory also accepts an external `device`, leaving it alive after teardown, and an injected asynchronous `read(url) -> ArrayBuffer`. `baseUrl` may name a precision directory directly. Only storage precision changes: activations and accumulation remain F32. FP16 uses core `unpack2x16float` unless native `shader-f16` is requested and supported; INT8 loads four signed bytes from each u32 and multiplies by row scales. Embeddings, geometry positional parameters, normalization parameters and biases remain F32. There are no additional FP16 stability exceptions.

`returnIntermediates: true` adds preprocessing tensors, graph, every encoder/decoder state (including edge states), head states, logits and outputs. Production calls omit these copies. Temporary buffers are pooled and reused within an inference, then all are destroyed in `finally`, including after callback exceptions and cancellation. `engine.memoryUsage()` reports resident/transient buffer counts and bytes; `destroy()` is idempotent and leaves totalBytes=0. Device loss, invalid inputs, overlapping predictions, missing/corrupt weights and allocation-limit failures reject. Aborting during command preparation stops encoding; submitted work finishes before readback cancellation is reported.

`webgpu/worker.js` is a real module-worker entry point. Send `{id, command:'init', options}`, `{id, command:'predict', text, options}`, `{command:'cancel'}`, or `{id, command:'destroy'}`. Replies have `{id,result}`, `{id,progress}`, or `{id,error,name}`. Output ArrayBuffers are transferred without copying. The demo exercises this protocol. `shared/` also exports `parseStructure`, `resolveMutations`, `bestMutants(result, {includeCys})`, per-chain backbone-atom `centrality` (Cα by default; absent basis atoms return -1, matching upstream), `toCSV`, and `toFASTA`. Iterate independent `predict` calls for a collection of structures.

## Reproducible export and numerical evidence

Run from this checkout. All installation caches and temporary files below stay inside the project:

```bash
mkdir -p .cache/tmp
UV_CACHE_DIR="$PWD/.cache/uv" uv venv .venv --python /usr/bin/python3.12
UV_CACHE_DIR="$PWD/.cache/uv" TMPDIR="$PWD/.cache/tmp" \
  uv pip install --python .venv/bin/python -r tools/requirements-export.txt
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/download_weights.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/make_inputs.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/export_thermompnn.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/dump_oracle.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/dump_topk.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/test_export.py
npm_config_cache="$PWD/.cache/npm" npm ci --ignore-scripts
```

Python 3.12.3, CPU PyTorch 2.5.1, NumPy 1.26.4; all installed export dependencies are pinned in `tools/requirements-export.txt`. Downloaded checkpoints are verified against hardcoded SHA-256 digests. Export removes optimizer state, uses deterministic safetensors serialization and reloads **every bundle into the original full upstream model**, including its unpruned convolution modules. `fixtures/export_metrics.json` records quantization error. The full original 20-mutations-per-site loop is also checked against the matrix extraction on the monomer.

Manifest format is **`thermompnn-safetensors-v1`**, with `model`, `variant`, `precision`, `upstream_commit`, `total_params`, `num_tensors`, `total_bytes`, `config`, per-shard `file`/`byteLength`/`sha256`, and a tensor-to-shard map. Tensor/parameter counts describe the logical inference graph; INT8 scale companions are additional raw entries. Each tensor gives `key`, `dtype`, `shape`, `row_start`, `shard`, or ordered row-split `parts`. I8 tensors use symmetric per-output-row quantization (`max(abs(row))/127`, zero-row floor 1e-30), round-to-nearest-even, and colocated F32 `<key>.__scale` tensors. Every serialized shard, **including its header**, is ≤25,000,000 bytes. Tests force the actual exporter to split at 8,192 bytes and round-trip all precisions. The loader validates sizes, SHA-256, tensor shapes, parts, colocated scales, and totals through Web Crypto; corruption is never accepted. Cache API/OPFS failures fall back to fetching, and incomplete cached manifests are refetched.

<!-- METRICS_START -->
| Bundle | Shards | Safetensors bytes | Weight relRMS | ΔΔG relRMS vs original FP32 (range) |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 1 | 7,356,136 | 0.000e+0 | 0.000e+0–0.000e+0 |
| FP16 | 1 | 3,715,934 | 2.065e-4 | 3.711e-4–5.881e-4 |
| INT8 | 1 | 1,933,201 | 1.047e-2 | 1.772e-2–4.657e-2 |

FP16 ddG quantization error is below 0.06% on these inputs. INT8 trades accuracy for download size (up to 4.66% relative ddG error here, maximum absolute deviation 0.201 kcal/mol on the full chain). These are measured storage effects; the arithmetic parity threshold remains 1e-4 for **every** precision. No biological accuracy claim is inferred from these fixtures.

Every row below passes SSM and targeted scoring, all intermediate traces, exact integer preprocessing/graphs, and WT identity checks. CPU numbers are the worst trace across all three storage precisions; GPU output columns compare against identically quantized PyTorch. “All GPU traces” includes native shader-f16.

| Input mode | CPU max trace relRMS | GPU FP32 ΔΔG | GPU FP16 ΔΔG | GPU INT8 ΔΔG | All GPU traces, max relRMS | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| monomer | 1.364e-6 | 5.888e-7 | 4.550e-7 | 5.598e-7 | 9.922e-7 | PASS |
| monomer_cif | 1.364e-6 | 5.888e-7 | 4.550e-7 | 5.598e-7 | 9.922e-7 | PASS |
| medium | 1.343e-6 | 8.344e-7 | 7.880e-7 | 7.208e-7 | 1.334e-6 | PASS |
| multichain | 1.681e-6 | 1.426e-6 | 1.298e-6 | 9.960e-7 | 2.044e-6 | PASS |
| multichain_cif | 1.681e-6 | 1.426e-6 | 1.298e-6 | 9.960e-7 | 2.044e-6 | PASS |
| chain_selection | 1.377e-6 | 9.722e-7 | 9.269e-7 | 8.224e-7 | 1.426e-6 | PASS |
| chain_selection_cif | 1.377e-6 | 9.722e-7 | 9.269e-7 | 8.224e-7 | 1.426e-6 | PASS |
| residue_edge_cases | 1.371e-6 | 4.285e-7 | 3.910e-7 | 4.882e-7 | 1.154e-6 | PASS |
| residue_edge_cases_cif | 1.371e-6 | 4.285e-7 | 3.910e-7 | 4.882e-7 | 1.154e-6 | PASS |

The additional complete **194-residue 2OCJ chain A** passes final-output parity (CPU max 6.470e-7, GPU max 7.335e-7), exercising dispatches beyond one workgroup-grid dimension. Full intermediate traces are checked on the mode fixtures above.

Measured on NVIDIA GB10, Chromium 153.0.8010.12:

| Storage / loads | Cold engine load (ms) | Warm 12 residues (ms) | Warm 76 residues (ms) | Warm 194 residues (ms) |
| --- | ---: | ---: | ---: | ---: |
| fp32 | 240.8 | 5.0 | 23.3 | 50.6 |
| fp16 | 258.2 | 3.4 | 20.6 | 49.6 |
| fp16-native | 258.0 | 4.2 | 19.5 | 54.4 |
| int8 | 265.2 | 4.5 | 20.0 | 53.4 |
<!-- METRICS_END -->

Fixtures derive from the upstream 2OCJ example: a 12-residue fragment, a 76-residue fragment, two 8-residue chains, chain selection, an 11-position parser stress case, and a complete 194-residue chain for final-output/latency validation. They test numerical fidelity, not experimental predictive accuracy. Little-endian F32/I32 oracle files include all layer states and quantized-reference traces; paired equivalent mmCIF cases reuse the corresponding PDB oracle. The medium trace is ~15.9 MB per precision; complete fixtures total ~56 MB. Tests require these deterministic artifacts and local bundles; missing files fail with no skips. No checkpoint, environment, or cache is part of the new port commit.

## Hardware verification

```bash
npm test                  # shared, vendor smoke, CPU, and hardware GPU
npm run test:cpu           # every trace / mode, all storage precisions
npm run test:gpu           # real Chromium WebGPU; software adapters fail
npm run demo              # prints an ephemeral loopback URL; open /demo/
```

Validated on ARM64 **NVIDIA GB10**, driver **580.159.03**, Node **22.21.1**. Chromium is `/home/ruh/.cache/ms-playwright/chromium-1243/chrome-linux-arm64/chrome` (`CHROME_PATH` can override). The harness uses a fresh profile for each run and **only an ephemeral port (`listen(0)`)**, isolates its profile/cache/temp files inside this checkout, selects `/usr/share/vulkan/icd.d/nvidia_icd.json`, and unsets DISPLAY/WAYLAND_DISPLAY. Playwright default launch arguments are disabled because they prevented hardware adapter creation on this build. Working GPU flags:

```text
--headless=new --no-sandbox --disable-gpu-sandbox --ozone-platform=headless
--enable-unsafe-webgpu --ignore-gpu-blocklist --enable-features=Vulkan
--use-angle=vulkan --use-vulkan=native --disable-vulkan-surface
--enable-dawn-features=vulkan_enable_f16_on_nvidia
```

The suite verifies FP32, core-unpacked FP16, native shader-f16 FP16, and packed INT8 against their matching upstream oracle with **relRMS < 1e-4 for every trace** and exact integer graph/token tensors. Quantization error is reported separately above. It tests both scoring modes for every input case, exact WT zeros, invalid-input rejection, transient/resident cleanup, pre/during-load/encoding cancellation, event-loop cancellation, injected callback errors, concurrent-call rejection, buffer-limit admission, caller device survival, and destruction during inference. All 36 adversarial PyTorch top-k tie-order cases (1–2,048 entries) pass on CPU and GPU.

A real module Web Worker loads weights and predicts with Cache API and OPFS after **browser networking is disabled**; typed result buffers transfer back to the page. Real browser storage methods are fault-injected with SecurityError/QuotaExceededError and inference still executes. The demo is tested for stylesheet loading, SSM, targeted scoring, precision switching, three consecutive CSV downloads with complete-file validation, and Release GPU. Per-trace results, lifecycle/storage results, adapter evidence, and warm timings are saved in `fixtures/cpu_metrics.json` and `fixtures/gpu_metrics.json`. Warm latency is wall time (encoding, execution and readback), median of five calls after warmup, without intermediate copies; cold load includes download, checksum validation and upload but excludes first-use shader compilation.

## Demo and vendor contract

`demo/` provides PDB/mmCIF upload/paste, chain selection, precision/cache controls, full saturation or targeted scoring, progress, cancellation, a residue-identified heatmap, best-mutant table with optional cysteine exclusion, CSV/FASTA/original-structure downloads, and explicit GPU release. Weights must be exported before using it. The example is deliberately a small fragment.

```bash
node sync-upstream.mjs .cache/vendor-smoke          # dry-run preview
node sync-upstream.mjs .cache/vendor-smoke --write  # local smoke destination
```

The sync script copies **only** `shared/`, `webgpu/`, optional `cpu/` (omit with `--no-cpu`), `LICENSE`, and this README. No weights, fixtures, Python, environments, demo, or build/test dependencies are copied. The smoke test imports the copied GPU entry and executes the copied CPU engine against an upstream oracle. To integrate elsewhere, point this script at the approved vendor directory; no integration repository is modified by this task.

Proposed `denovo.run` ports:

| Port | Contract |
| --- | --- |
| `StructureArtifact` input | PDB/mmCIF text plus selected chain IDs; preserve author residue identities |
| `SequenceSet` input | Optional mutant sequences: an adapter validates one substitution relative to the structure WT sequence and converts it to a targeted descriptor; no independent sequence-only prediction |
| `ResidueTrack` output | WT sequence, geometry mask, valid-site mask, L×20 ΔΔG values and canonical alphabet |
| `Collection` output | Targeted mutation records `{chain,number,ins,index,wildtype,mutation,ddG}` or best-per-site records |
| `StructureArtifact` / `SequenceSet` download | Original uploaded structure and extracted WT FASTA; no generated coordinates |
| Worker transport | Transfer `ddg`, `scores`, `mask`, `valid` ArrayBuffers; ordinary residue metadata stays structured-clone data |

The browser package exports `thermompnn-webgpu`, `thermompnn-webgpu/webgpu`, `thermompnn-webgpu/cpu`, and `thermompnn-webgpu/shared`. The original upstream README and Python training/evaluation utilities remain as provenance; browser users should follow this file.
