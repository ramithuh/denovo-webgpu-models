# Dynamic BoltzGen for JavaScript and WebGPU

BoltzGen-1 coordinate inference with dynamic protein lengths, diffusion steps, and recycling. Neural operations use JavaScript on CPU or WGSL on WebGPU. There is no captured operation graph, Python service, ONNX, WASM, or CPU neural fallback in the WebGPU engine.

Pinned upstream: [HannesStark/boltzgen, 6472b44](https://github.com/HannesStark/boltzgen/tree/6472b44e8072187e6eaa51271ae6838245206013). Checkpoints use Hugging Face revision `c1be29e1f82ffcc72264f64b993c43fb4e0d17f0`. Both `boltzgen1_diverse` and `boltzgen1_adherence` share the supported architecture. Weight provenance and schema are recorded in `fixtures/architecture.json`.

## Generate coordinates

```js
import { createBoltzGenEngine } from './webgpu/index.js';

const engine = await createBoltzGenEngine({
  baseUrl: './weights/diverse/fp16/',
  precision: 'fp16',
  cache: 'cache', // 'none', 'cache', or 'opfs'
  // device: callerOwnedGPUDevice,
});
try {
  const result = await engine.generate({
    mode: 'unconditional',
    L: 32,
    numSteps: 20,
    recyclingSteps: 0,
    seed: 42,
    // signal: abortController.signal,
    // onProgress: ({ phase, block, step, total, recycle }) => { ... },
  });
  console.log(result.atom_coords, result.atoms, result.trajectory);
} finally {
  engine.destroy();
}
```

`createBoltzGenCPUEngine` in `cpu/index.js` has the same API. Node callers supply `read(path): Promise<ArrayBuffer>`; runtime modules contain no Node dependencies. CPU inference is a numerical reference and is substantially slower than hardware WebGPU.

`L` is a positive integer, with no fixed trace length. `numSteps` is a positive integer (default 3); `steps` remains an alias. `recyclingSteps` is a nonnegative integer (default 0); there are `recyclingSteps + 1` trunk passes. A one-step schedule uses the maximum sigma followed by zero; the upstream formula is undefined for one step. Device buffer limits and available memory bound practical input sizes. Pair features scale quadratically and triangle attention cubically with total token count.

Outputs contain flattened `[nAtoms,3]` Float32 coordinates, one denoised tensor per step, and `numSteps + 2` trajectory tensors. The trajectory includes the initial noise and upstream's duplicated final entry. Window padding is removed from public outputs. Every designed residue has 14 atom slots, including slots that may be unused by a subsequently assigned sequence. `atoms` identifies each slot's token, chain, residue, atom name, and design status.

## Targets and motifs

```js
// A separate designed chain and an arbitrary protein target.
const binder = await engine.generate({
  mode: 'binder', L: 16, numSteps: 50, seed: 7,
  targetPDB: pdbText,
});

// Alternatively supply a standard one-letter sequence and flattened or nested
// N, CA, C, O coordinates for every target residue.
const backbone = await engine.generate({
  mode: 'binder', L: 16, numSteps: 20,
  targetSequence: 'ACDEFGHIKL',
  targetCoords: coordinates, // 10 × 4 × 3 values
});

// Place fixed motif residues within a scaffold of total length L.
const scaffold = await engine.generate({
  mode: 'motif', L: 32, numSteps: 20, recyclingSteps: 1,
  targetPDB: motifPDB,
  motifIndices: [5, 6, 7, 8], // zero-based; one per motif residue
});
```

The PDB reader uses the first model, blank/A alternate locations, residue insertion codes, and multiple chains. All 20 standard amino acids are supported. Every target residue requires N, CA, C, O; supplied canonical sidechain atoms are retained. Hydrogen and unsupported atom names are ignored; unsupported residue types and missing backbone atoms fail explicitly. Binder output metadata assigns the designed chain A and target chains B onward, retaining each target's `sourceChain`.

Featurization builds atom-to-token maps, local attention windows, masks, residue/profile encodings, chain/entity relationships, distance conditioning, and empty template/MSA defaults at the requested shape. CCD conformers in `shared/residues.js` are chemical input data exported from pinned inference-data, not predictions. Reference conformers receive seeded rigid augmentation, preserving upstream's last-UID exception. Designed sidechains use masked element/name features; designed backbone atom identities remain visible as in the checkpoint's masker.

Targets and motifs are conditioning inputs: the sampler predicts their coordinates too, following upstream. It does not clamp target coordinates after each step. This API produces coordinates; residue assignment from atom14 geometry, inverse folding, confidence refolding, ranking, and biological validation remain downstream work. RNA, DNA, ligands, general upstream YAML/mmCIF, and supplied MSAs are outside the public input builder. The lower-level `run({features, random, numSteps, recyclingSteps})` accepts complete feature tensors for numerical validation and advanced integrations.

## Modules and execution

- `shared/input.js`, `residues.js`, `random.js`: dynamic protein input construction, chemical metadata, geometry, seeded draws, sigma schedule.
- `shared/encoders.js`: InputEmbedder, AtomEncoder, three-layer AtomAttentionEncoder/Decoder, residue and design conditioning, DiffusionConditioning with 24 token-layer biases.
- `shared/layers.js`, `trunk.js`: outgoing/incoming triangle multiplication, starting/ending triangle attention, token attention, SwiGLU transitions, two distance blocks, two template blocks, four MSA blocks, and 64 Pairformer blocks.
- `shared/model.js`: trunk recycling and the score network, including Fourier time conditioning, three atom encoder layers, 24 DiffusionTransformer layers, three atom decoder layers, and coordinate projection.
- `shared/sampler.js`: reverse EDM/AF3 loop, centering, quaternion rigid augmentation, noise injection, preconditioning, Kabsch alignment through a 3×3 SVD, and Euler updates. Upstream's next-sigma churn choice and noise-scale semantics are preserved. The checkpoint uses noise/step scales of 1.
- `cpu/backend.js`: JavaScript tensor kernels. `webgpu/backend.js` and `shaders.js`: GPU storage, scheduling, and WGSL tensor kernels. `shared/ops.js` is eager tensor algebra and scoped resource management; it does not record or interpret a program.

Distogram and B-factor heads are not needed to generate coordinates and are not exposed by this API. Weights for the complete checkpoint are retained for schema compatibility.

## Packed GPU weights and lifecycle

All six exports retain verified safetensors shards smaller than 25 MB. FP16 matrices remain packed two values per word; INT8 matrices remain packed four values per word with per-output-row scales. WGSL decodes weights where they are consumed and accumulates in FP32. Embeddings, Fourier parameters, and vectors retain FP32 storage. No full-size FP32 shadow weight copy is allocated on GPU.

| Precision | Resident weight bytes |
| --- | ---: |
| FP32 | 1,985,992,584 |
| FP16 | 994,301,256 |
| INT8 | 502,533,780 |

The engine frees transient buffers after success, failure, or cancellation. `destroy()` is idempotent, cancels active work, and frees weights, scratch buffers, and cached pipelines. Caller-provided GPU devices remain caller-owned. Concurrent inference on one engine is rejected; use separate engines when needed. Shader pipeline caching is bounded. The module worker supports `init`, `run`, `generate`, `abort` (with a target request ID), and `destroy`.

WGSL uses mutable function/workgroup arrays where indexing is dynamic; no dynamically indexed `const array` is generated. All 1,806 emitted shader variants for FP32/FP16/INT8 at lengths 8, 20, 24, and 64 also compile through wgpu-native/Naga on NVIDIA GB10 with zero failures (`fixtures/naga_validation.json`). This exercises the Naga compiler used by the Firefox WebGPU stack; full browser inference is measured in Chromium. Chromium hardware WebGPU is the measured validation platform; browser availability is checked explicitly and software adapters fail instead of skipping tests.

## Numerical validation

`tools/dump_oracle.py` runs the unmodified upstream model on exact feature tensors and seeded random draws exported by `tools/prepare_inputs.mjs`. It saves outputs and module checkpoints, never ATen graphs. FP16/INT8 references use identically decoded weights with FP32 PyTorch arithmetic. This measures engine parity separately from quantization drift against the original FP32 checkpoint.

The matrix covers diverse FP32/FP16/INT8 and adherence FP32, each with L=8/3 steps, L=24/5 steps, and an L=8 binder/3 steps with 12 real non-glycine crambin residues. A diverse FP16 motif case adds a 16-residue scaffold, five steps, and one recycle. The target snippet is [RCSB PDB 1CRN](https://files.rcsb.org/download/1CRN.pdb), chain A residues 1–12, saved in `fixtures/1crn-12.pdb`.

Every real atom, each denoised step, each reverse update, and final coordinates must have `relRMS < 1e-4`, with **no alignment fitted during evaluation**. Alignment inside the sampler is part of BoltzGen itself. The GPU matrix checks both explicit tensor inputs and public `generate()` results. Node tests exercise public CPU generation. Hardware tests cover buffer cleanup on success/error/abort, external device ownership, worker inference, Cache/OPFS offline hits, storage-denial fallback, and the interactive demo. Worker smoke cases also exercise L=16/10 steps, L=32/20 steps, and L=64/50 steps; these longer cases test execution and finiteness rather than biological quality.

Measured records are in `fixtures/cpu_*`, `fixtures/gpu_*`, `fixtures/browser_contracts.json`, and `fixtures/validation.json`. `fixtures/quantization_drift.json` compares quantized PyTorch coordinates against FP32 separately. INT8 numerical parity with its own reference does not establish equivalence to FP32 design quality.

## Reproduce

```sh
cd models/boltzgen
# Existing environment and exported weights can be reused.
node tools/prepare_inputs.mjs
.venv/bin/python tools/dump_oracle.py --precision fp32
.venv/bin/python tools/dump_oracle.py --precision fp16
.venv/bin/python tools/dump_oracle.py --precision int8
.venv/bin/python tools/dump_oracle.py --variant adherence
npm test
npm run test:cpu
npm run test:gpu
.venv/bin/python tools/summarize.py
.venv/bin/python tools/quantization_metrics.py
# Optional independent Naga validation (Python tooling only):
uv pip install --python .venv/bin/python --cache-dir .cache/uv -r tools/requirements-naga.txt
node tools/export_shaders.mjs
.venv/bin/python tools/validate_wgsl.py
node tools/serve.mjs 8080
# http://127.0.0.1:8080/demo/
```

The demo lets you choose checkpoint, precision, design length, diffusion steps, recycling, mode, seed, and target PDB; rotate the atom view; cancel; download coordinates; and release GPU resources. Full cold environment/export setup remains in `tools/bootstrap.sh`. Browser validation uses the supplied ARM64 Chromium binary, configurable with `CHROME_PATH`, and Vulkan on NVIDIA GB10. Missing fixtures, corrupt shards, missing hardware, numerical failures, and shader errors are failures, never skips.

Measured acceptance run: `npm test` **13/13**, `npm run test:cpu` **4/4**, `npm run test:gpu` **2/2**, all with **0 failures and 0 skips**. Command summaries are saved in `fixtures/test_results.json`. Across all 13 parity scenarios per backend, the worst checked tensor relRMS is **3.131e-5 CPU** and **2.988e-5 GPU**.
