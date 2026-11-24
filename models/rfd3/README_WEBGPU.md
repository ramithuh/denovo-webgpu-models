# RFD3 JavaScript and WebGPU browser port

A complete forward implementation of the pinned RFdiffusion3 checkpoint, with pure JavaScript CPU and pure WGSL WebGPU engines. The GPU path runs the token initializer, pairformer conditioning, atom encoder, all 18 token-transformer blocks, block-streamed atom decoder, coordinate projection, sequence head, and recycling. Neural activations stay on the GPU until requested outputs are read back. There is no CPU neural fallback, ONNX, WASM, Python service, or captured execution trace.

The browser also implements input construction, AF3/EDM sampling, shared random draws, Kabsch/SVD augmentation, symmetry, conditioning, and PDB/mmCIF export. Offline Python tools produce weights and numerical oracles; they are not runtime dependencies. FP32 is the numerical reference. FP16 is validated for generation. **INT8 remains experimental:** its kernels agree with quantized PyTorch, but quantizing the model changes coordinates and sequences. The geometry measurements below include stricter van der Waals overlaps; these outputs are not universally clash-free.

## Provenance and packed weights

- Foundry commit: `a13b299a05897b951ea28b65e977eebb984380bc`.
- Official checkpoint: `/home/ruh/.foundry/checkpoints/rfd3_latest.ckpt`.
- SHA-256: `9b3f85923e0d51e9453e15cdd2f8c666e7ce096a60577f57d11bbc54ae6d67c1`.
- Inference uses the EMA `shadow` branch, matching upstream evaluation dispatch.
- **168,038,994 learned parameters**, 168,040,050 state elements, 807 state tensors. The boolean output mask is a manifest constant; 806 floating tensors are sharded.
- [Architecture and tensor schema](fixtures/architecture.json), [independent export verification](fixtures/export-verification.json).
- [BSD-3-Clause](LICENSE), upstream kernel [NOTICE](NOTICE), [Apache-2.0](LICENSE-APACHE-2.0), and [PyTorch sorting provenance/license](LICENSE-PYTORCH) are distributed with the port.

| Bundle | Shards | Total shard bytes | Largest shard | Packed GPU weight bytes |
|---|---:|---:|---:|---:|
| FP32 | 29 | 672,279,616 | 24,932,448 | 672,160,072 |
| FP16 | 14 | 336,887,416 | 24,856,952 | 336,768,744 |
| INT8 | 7 | 170,394,484 | 24,710,784 | 170,199,268 |

Every shard is **strictly smaller than 25,000,000 bytes**. SHA-256 and length are verified on both fresh and cached reads. All bundles remain under the authorized `weights/` symlink; no multi-GB copy is stored on `/`.

Linear matrices use FP32, packed FP16 (two values/u32), or packed signed INT8 (four values/u32 with per-output-row FP32 scales). WGSL decodes in the matrix kernel and accumulates in FP32. **There is no full-size FP32 GPU weight shadow.** Embeddings, norm gains, Fourier parameters, biases, and other vectors stay FP32. The CPU reference may expand weights in CPU memory.

## Verified inputs and behavior

The same `buildInput()` and high-level `generate()` API serve both engines. Feature tests compare every tensor against the pinned upstream pipeline, including chemical identities, representative atoms, reference masks, token bonds, positional encoding, and conditioning masks.

| Mode / feature | Verification and scope |
|---|---|
| Unconditional monomer | Feature and denoiser parity; real 64-residue, 50-update generation in all precisions |
| Indexed atomic motif | N/CA and backbone selections; HIS/SER/ASP atomic-site fixture; held atoms preserved |
| Unindexed motif | Feature/denoiser parity; guidepost-to-design assignment, optional cleanup, pre-insertion RMSD metadata |
| Binder / hotspots | Separate target chain and atom-level hotspots; feature/denoiser parity |
| Small molecule | ATP atomization, chemical reference masks/elements/charges, and covalent token bonds; full FP16 trajectory |
| DNA / RNA | Canonical nucleotide tokenization and C2/C4 representatives; DA and A oracle fixtures |
| Cyclic peptide | One complete de novo chain, matching upstream scope; 12-residue trajectory and closure measurements |
| H-bond / exposure | Donor, acceptor, buried, partially buried, exposed tensors; oracle and trajectory tests |
| Fixed sequence / sidechain-only | Input residue identity stays fixed; `BKBN` fixes N/CA/C/O while other coordinates diffuse |
| Fixed backbone with sequence redesign | Whole-residue sequence unmasking and Atom14 padding; oracle fixture |
| Partial diffusion | Source coordinates, filtered schedule, and mandatory >=2 remaining points; oracle fixture |
| Rigid augmentation | 3×3 Jacobi SVD/Kabsch; full-rank, planar, reflection, weighted and degenerate tests; shared-draw PyTorch trajectory |
| Symmetry | Atomwise ASU/entity/frame semantics; C2 high-level features and shared-draw trajectory; explicit frames at low level |
| CFG | Every `cfgScale !== 1`, including values below one; explicit reference features; incompatible symmetry requests rejected |
| Multichain neighbors | Four-chain, 32-token/448-atom oracle; reserved interchain selection and small top-k tie behavior |
| Dynamic execution | Variable atoms/tokens, batch dimension, diffusion steps, and recycling; 1/2/3-recycle numerical checks |

A conditioning feature being supported does not establish that a generated structure satisfies a biochemical function. H-bond formation, exposure adherence, binder affinity, and enzyme activity are not validated by these tests.

## Measured NVIDIA GB10 results

Measured in real headless Chromium ARM64 with the NVIDIA Vulkan ICD. PyTorch 2.10.0+cu130 oracles use eager FP32, TF32 disabled, and the upstream forced-BF16 pairformer path disabled. Repeated oracle denoisers are bit-identical. Cross-backend results are **tolerance-based numerical parity, not bitwise equality**.

<!-- MEASUREMENTS -->
Across 14 featurizer cases, the worst tensor maxAbsDiff is **0**, worst RMSE is **0**, and minimum cosineSim is **1.00000000000000**. [Per-tensor feature metrics](fixtures/featurizer-parity.json).

| GB10 single denoiser vs PyTorch FP32 (worst across conditioned cases) | maxAbsDiff Å | max RMSE Å | min cosineSim |
|---|---:|---:|---:|
| fp32 | 1.907349e-5 | 6.274381e-6 | 0.99999999999569 |
| fp16 | 4.531860e-3 | 1.406041e-3 | 0.99999978477657 |
| int8 | 4.487474e-1 | 1.237955e-1 | 0.99860057616421 |

| FP32 conditioned denoiser | GPU maxAbsDiff Å | GPU RMSE Å | GPU cosineSim | CPU maxAbsDiff Å |
|---|---:|---:|---:|---:|
| monomer | 1.382828e-5 | 4.502646e-6 | 0.99999999999702 | 1.049042e-5 |
| cyclic | 1.120567e-5 | 3.148444e-6 | 0.99999999999938 | 9.536743e-6 |
| atomic-motif | 1.430511e-5 | 6.274381e-6 | 0.99999999999569 | 1.049042e-5 |
| ligand | 1.025200e-5 | 2.459774e-6 | 0.99999999999971 | 1.907349e-5 |
| dna | 1.358986e-5 | 2.633957e-6 | 0.99999999999967 | 9.250641e-5 |
| binder-hotspot | 1.072884e-5 | 3.795420e-6 | 0.99999999999911 | 6.675720e-5 |
| unindexed | 9.298325e-6 | 2.961185e-6 | 0.99999999999908 | 1.621246e-5 |
| hbond-exposure | 1.525879e-5 | 5.346429e-6 | 0.99999999999796 | 1.525879e-5 |
| multichain | 1.907349e-5 | 3.972354e-6 | 0.99999999999887 | 2.098083e-5 |
| fixed-backbone | 5.960464e-7 | 1.546661e-7 | 0.99999999999999 | 1.430511e-6 |
| sequence-redesign | 1.049042e-5 | 2.523089e-6 | 0.99999999999773 | 7.629395e-6 |
| partial | 9.059906e-6 | 4.215359e-6 | 0.99999999999982 | 9.536743e-6 |
| rna | 1.192093e-5 | 3.553402e-6 | 0.99999999999937 | 3.767014e-4 |
| atomic-triad | 1.096725e-5 | 3.830136e-6 | 0.99999999999901 | 1.907349e-5 |

GPU vs JavaScript CPU FP32 coordinates: worst maxAbsDiff **3.709793e-4 Å**, worst RMSE **1.555093e-4 Å**, minimum cosineSim **0.99999999903042**.

FP32 GPU initializer worst maxAbsDiff: **4.577637e-5**; sequence logits: **2.502441e-3**. CPU sequence-logit worst maxAbsDiff: **5.861664e-2**. Raw logits have much larger magnitudes than coordinates. FP32 decoded indices are asserted identical to PyTorch.

| Packed full model vs PyTorch with the same quantized weights | Worst coordinate maxAbsDiff Å | Worst logit maxAbsDiff |
|---|---:|---:|
| fp16 | 2.741814e-5 | 2.685547e-3 |
| int8 | 1.749992e-4 | 3.613281e-2 |

| Shared-draw trajectory vs PyTorch FP32 (four schedule points) | Precision | maxAbsDiff Å | RMSE Å | cosineSim |
|---|---|---:|---:|---:|
| monomer | fp32 | 8.544922e-4 | 1.310068e-4 | 0.99999999999996 |
| aligned-motif | fp32 | 4.013062e-3 | 1.008574e-3 | 0.99999999999817 |
| symmetry | fp32 | 5.493164e-4 | 1.235512e-4 | 0.99999999999997 |
| monomer | fp16 | 1.141724e+0 | 3.776494e-1 | 0.99999972168644 |
| aligned-motif | fp16 | 4.598999e-1 | 1.693697e-1 | 0.99999995042615 |
| symmetry | fp16 | 9.997559e-1 | 3.854468e-1 | 0.99999971710079 |
| monomer | int8 | 5.382568e+0 | 2.178568e+0 | 0.99999051784368 |
| aligned-motif | int8 | 2.940994e+0 | 9.314325e-1 | 0.99999845291113 |
| symmetry | int8 | 8.079590e+0 | 2.600199e+0 | 0.99998713183795 |

These deliberately short, high-noise trajectories test arithmetic, not protein quality. Quantized trajectories are not expected to match FP32 coordinates exactly. Real generation below uses 20–50 updates.

| GB10 generation | Precision | Tokens / atoms | Updates | Warm step mean ms | Trajectory s | Peak GPU buffers GB |
|---|---|---:|---:|---:|---:|---:|
| monomer | fp32 | 64 / 896 | 50 | 104.10 | 8.322 | 1.782 |
| motif | fp32 | 64 / 887 | 50 | 110.30 | 7.606 | 1.763 |
| monomer | fp16 | 64 / 896 | 50 | 99.23 | 5.594 | 1.447 |
| motif | fp16 | 64 / 887 | 50 | 99.00 | 5.397 | 1.427 |
| ligand | fp16 | 55 / 367 | 20 | 81.03 | 2.840 | 0.608 |
| dna | fp16 | 25 / 358 | 20 | 62.03 | 2.486 | 0.530 |
| cyclic | fp16 | 12 / 168 | 30 | 67.33 | 3.633 | 0.386 |
| hbond-exposure | fp16 | 24 / 327 | 20 | 83.93 | 2.582 | 0.501 |
| unindexed | fp16 | 25 / 338 | 20 | 83.63 | 2.373 | 0.513 |
| binder | fp16 | 25 / 341 | 20 | 63.57 | 2.332 | 0.515 |
| rna | fp16 | 25 / 359 | 20 | 74.43 | 2.479 | 0.531 |
| realignment | fp16 | 24 / 327 | 20 | 68.03 | 1.468 | 0.501 |
| partial | fp16 | 1 / 5 | 5 | 66.40 | 0.530 | 0.337 |
| symmetry | fp16 | 24 / 336 | 20 | 69.17 | 2.236 | 0.509 |
| monomer | int8 | 64 / 896 | 50 | 100.83 | 5.540 | 1.280 |
| motif | int8 | 64 / 887 | 50 | 100.33 | 5.740 | 1.261 |

| Geometry (seed 20261123) | Precision | Protein residues | Breaks | Cα mean Å | Severe overlaps | vdW overlaps | Motif RMSD Å | PSEA helix / strand / coil |
|---|---|---:|---:|---:|---:|---:|---:|---|
| monomer | fp32 | 64 | 0 | 3.8166 | 0 | 4 | — | 0 / 28 / 36 |
| motif | fp32 | 64 | 0 | 3.7817 | 0 | 12 | 0 | 0 / 44 / 20 |
| monomer | fp16 | 64 | 0 | 3.8181 | 0 | 7 | — | 0 / 28 / 36 |
| motif | fp16 | 64 | 0 | 3.7816 | 0 | 11 | 0 | 0 / 44 / 20 |
| ligand | fp16 | 24 | 0 | 3.8044 | 0 | 5 | 0 | 15 / 0 / 9 |
| dna | fp16 | 24 | 0 | 3.7786 | 0 | 7 | 0 | 14 / 0 / 10 |
| cyclic | fp16 | 12 | 0 | 3.8290 | 0 | 0 | — | 0 / 0 / 12 |
| hbond-exposure | fp16 | 24 | 0 | 3.8860 | 0 | 0 | 0 | 22 / 0 / 2 |
| unindexed | fp16 | 24 | 0 | 3.7678 | 0 | 3 | 0 | 7 / 10 / 7 |
| binder | fp16 | 25 | 0 | 3.7719 | 0 | 3 | 0 | 0 / 6 / 19 |
| rna | fp16 | 24 | 0 | 3.7728 | 0 | 3 | 0 | 0 / 11 / 13 |
| realignment | fp16 | 24 | 0 | 3.8098 | 0 | 1 | 1.633622e-7 | 7 / 0 / 17 |
| partial | fp16 | 1 | 0 | — | 0 | 0 | — | 0 / 0 / 1 |
| symmetry | fp16 | 24 | 0 | 3.8225 | 0 | 3 | — | 0 / 0 / 24 |
| monomer | int8 | 64 | 0 | 3.8155 | 0 | 3 | — | 0 / 28 / 36 |
| motif | int8 | 64 | 0 | 3.7810 | 0 | 12 | 0 | 0 / 44 / 20 |

Cyclic closing C–N: **1.3265 Å**; closing Cα–Cα: **3.8424 Å**.
Unindexed pre-insertion motif RMSD: **0.758970 Å**. Export cleanup inserts the fixed atoms; the largest resulting peptide C–N distance is **1.5793 Å**. This is reported separately from raw guidepost preservation.
<!-- /MEASUREMENTS -->

Warm step times include submission, completion, and output readback after warm-up (three measurements, two recycles). Trajectory times include initialization and updates, exclude weight download/upload, and benefit from already-compiled pipelines. Cold initialization/compilation is separate. Buffer peaks count live GPUBuffer backing allocations, including scratch pools and packed weights; **they are not total driver VRAM measurements**. GB10 is a unified-memory device. No claim is made about other hardware.

Geometry is measured on exported coordinates with Biopython/Biotite, including the explicitly documented insertion of unindexed motif atoms. A chain break is adjacent Cα distance >4.5 Å. “Severe overlaps” means nonbonded heavy-atom distance <1.5 Å; the separate van der Waals column counts overlap >0.4 Å using the documented radii in `tools/analyze_geometry.py`. Both exclude covalent 1–2, 1–3 and 1–4 pairs, including cyclic closure. Secondary structure uses Biotite PSEA, not DSSP. One seed per benchmark is insufficient to establish a distribution-wide quality guarantee.

Raw evidence: [GPU parity](fixtures/gpu-model-parity.json), [CPU parity](fixtures/cpu-denoiser-parity.json), [generated structures/timings](fixtures/gpu-generation.json), [independent geometry](fixtures/generation-geometry.json), [demo verification](fixtures/demo-verification.json), [base oracle](fixtures/oracle/manifest.json), [conditioned oracle](fixtures/oracle-pass2/manifest.json), [alignment](fixtures/alignment-oracle.json), [sampler](fixtures/sampler-oracle.json).

### INT8 investigation

The Phase 1 ~0.35 isolated-layer error was weight quantization error, not incorrect packed decoding. Full-model PyTorch with the same rowwise INT8 transformation reproduces the larger coordinate/logit drift. Promoting **both** the sequence head and final coordinate projection to FP32 improves the monomer single-step coordinate max error from approximately 0.1401 to 0.0780 Å, and logit error from 60.58 to 35.06. It does not restore FP32 behavior because the hidden activations are already changed by quantization. The shipped INT8 bundle therefore remains the explicitly labeled baseline, without silently substituting mixed weights. See [quantization analysis](fixtures/quantization-analysis.json) and [quantized PyTorch outputs](fixtures/quantized-oracle.json).

Sequence entropy uses softmax over all 32 raw logits, as upstream does. Decoding applies softmax, the checkpoint mask, and first-index tie semantics, including all-zero probabilities. WGSL explicitly preserves FP32 subnormal ordering for this decoder because WebGPU permits flushing subnormal arithmetic; replacing it with a masked logit argmax changes ligand-associated predictions.

## Public API

```js
import { createRFD3Engine } from './webgpu/index.js';
// Alternatively: import { createRFD3CPUEngine } from './cpu/index.js';

const baseURL = new URL('./weights/fp16/', location.href);
const manifest = await (await fetch(new URL('manifest.json', baseURL))).json();
const architecture = await (await fetch('./fixtures/architecture.json')).json();
const engine = await createRFD3Engine({ manifest, architecture, baseURL });
try {
  const design = await engine.generate({
    mode: 'unconditional', length: 64,
    steps: 51,       // 51 schedule points = 50 Euler updates
    n_recycle: 2, seed: 20261123,
    retainTrajectory: false,
    onStep: ({ step, total }) => console.log(step + 1, total),
  });
  console.log(design.seed, design.pdb, design.mmcif);
  // Persist the returned seed, including when the engine chose it automatically.
  console.log(engine.peakMemoryUsage());
} finally {
  engine.destroy();
}
```

CPU `initialize`, denoiser `run`, and `sample` are synchronous; use a Worker to avoid blocking the UI. CPU `generate` returns a Promise. GPU methods are asynchronous. Both return `{shape, data}` tensors named `X_L`, `sequence_logits_I`, and `sequence_indices_I`. Sampling additionally returns the resolved seed, schedule, entropy trajectory, and optionally coordinate trajectory. High-level generation adds `rows`, `pdb`, `mmcif`, `features`, and the source-frame `origin`.

### Conditioning examples

`structure` is PDB/mmCIF **text**, not a path or a URL. `contig` uses concrete lengths, source residue selections, and `,/0,` for a chain break. `fixedAtoms` maps source residue IDs to comma-separated atom names; `BKBN` means N/CA/C/O. Source residue identity is fixed by default.

```js
const motif = await engine.generate({
  mode: 'motif', structure: pdbText,
  contig: '30,A1,33', fixedAtoms: { A1: 'N,CA,C,O' },
  steps: 51, seed: 1,
});
const unindexed = await engine.generate({
  mode: 'motif', structure: pdbText, length: 64,
  unindex: 'A1', fixedAtoms: { A1: 'N,CA' }, steps: 51, seed: 1,
});
const binder = await engine.generate({
  mode: 'binder', structure: targetPdbText, length: 64,
  target: 'A1-30', hotspots: { A12: 'CA,CB' }, steps: 51, seed: 1,
});
const ligand = await engine.generate({
  mode: 'conditioned', structure: ligandPdbText,
  length: 64, ligand: 'ATP', steps: 51, seed: 1,
});
const dna = await engine.generate({
  mode: 'conditioned', structure: dnaCifText,
  contig: '64,/0,D1-8', steps: 51, seed: 1,
});
```

Additional options:

- `cyclic: true`: one complete de novo peptide, at least three residues.
- `hbondDonor`, `hbondAcceptor`, `hotspots`: residue/atom selections. Upstream aliases `select_hbond_donor`, `select_hbond_acceptor`, `select_hotspots` are accepted.
- `select_buried`, `select_partially_buried`, `select_exposed`: atom selections. **Pinned upstream behavior:** later selections reset the earlier RASA annotation; the order is buried → partially buried → exposed. For simultaneous independent assignments, construct `features.ref_atomwise_rasa` explicitly.
- `unfixedSequence: true` (or a residue selection): redesign whole protein residues while retaining chosen fixed coordinates; Atom14 padding is generated. Partial atom-level sequence unmasking is rejected.
- `partialT`: positive noise cutoff. Provide complete initial source coordinates with `contig` and the desired `fixedAtoms` mask, or explicit flat `coordinates` in the model-centered frame (`source coordinates - buildInput(options).origin`). At least two schedule points must remain. Example: `{structure, contig:'A1-64', fixedAtoms:false, partialT:5, steps:51}`.
- `allowRealignment: true`, `centerOption: 'all' | 'motif' | 'diffuse'`, `sTrans`: upstream rigid augmentation and final motif alignment.
- `symmetry: {id:'C2'}` with de novo `length`: length per asymmetric unit. Other explicit rigid frames are accepted through low-level symmetry features. Symmetry requires `gamma0 > 0.5` and `cfgScale === 1`.
- `cleanupGuideposts` defaults to true: assign unindexed atoms to generated residues, record pre-insertion RMSD in `motifAssignments`, insert held atoms, and remove guidepost tokens from exports. Low-level `X_L` remains the raw neural/sampler output. Missing feasible atom-name assignments fail explicitly. `false` exports the raw guideposts.
- `signal`: AbortSignal checked during loading and between diffusion updates. GPU commands already submitted finish before cancellation is observed.

The canonical reference library contains A/C/G/U, DA/DC/DG/DT, ATP and zinc. `buildReferenceConformer(name)` returns independent atom/bond/coordinate records; `components: [{name, atoms, bonds}]` supports explicitly supplied chemistry for other small molecules. This is not a browser SMILES/CCD downloader or RDKit conformer generator. The pinned checkpoint masks reference positions to zero; reference masks, elements, charges, names and bonds still matter and are verified. Protein atom ordering comes from the pinned RFD3 dense association scheme.

### Deterministic low-level API

```js
import { buildInput } from './shared/input.js';
const { features } = buildInput({ length: 4 });
const initializer = await engine.initialize(features);
try {
  const output = await engine.run({
    features, initializer,
    X_noisy_L: explicitCoordinates, // Float32Array [batch, atoms, 3]
    t: 16, n_recycle: 2,
  });
  const trajectory = await engine.run({
    features, schedule: explicitSchedule,
    random: { initial: initialNormals, steps: updateNormals },
    seed: 0, n_recycle: 2,
  });
} finally {
  engine.releaseInitializer?.(initializer);
}
```

`run` with `X_noisy_L` is one denoiser call. `run` with `random` and no noisy coordinates delegates to `sample`. For realignment parity, shared randomness also needs `augmentations: [{angles, translation}, ...]` and `finalAugmentation` when a fixed motif exists; angles are radians. An explicit schedule must be descending, finite and positive, with at least two points. `cfgScale !== 1` requires explicit `referenceFeatures` with a compatible indexed prefix. GPU `readInitializer(handle)` exposes the five initializer tensors for diagnostics; `inspectNeighbours` exposes computed neighbor multiplicities.

Initializers belong to their engine and feature object. Release handles when finished. `trim()` releases idle pooled scratch buffers and resets peak accounting to current residency; `destroy()` releases all weights/activations. Do not concurrently mutate/release features or handles used by an active operation. Device loss invalidates the resource owner and zeros live accounting. GPU validation/allocation failures close the engine rather than reusing invalid resources.

## Memory, structure I/O, and integration

Adapter storage/backing-buffer limits are negotiated independently. Matrix dispatches span multiple dimensions when a grid exceeds 65,535 groups. Decoder and recycle temporaries are recycled after their final encoded use. Pair activations are still dense: peak memory grows quadratically with atoms, and tensors exceeding the negotiated binding limit are explicitly rejected. This implementation does not offer arbitrary-length, out-of-core pair tiling. The measured 64-residue cases fit within 1.3–1.8 GB of tracked buffers; the demo caps length at 100.

`shared/structure.js` preserves PDB atom alignment, checks identity collisions and author numbering, emits standard PDBx tag capitalization, groups models contiguously, and preserves globally unique atom IDs. CIF tokenization supports quoted missing-value literals, closing-semicolon trailing tokens and numeric uncertainties, while rejecting JavaScript hexadecimal syntax. Coordinate conversion preserves rows, not every assembly/bond annotation or CIF category. Cyclic closure is modeled and measured, but these coordinate exports do not encode a complete bond topology.

Generated output is independently parsed by Biopython and **ProteinDesignSpace's actual TypeScript parser**, compiled in memory from the supplied checkout. Every generated atom is retained, including ATP and DNA. ProteinDesignSpace's `nResidues` counts protein residues; DNA remains in its coordinate records.

High-level input construction currently requires a single selected model, canonical protein/nucleic-acid residues or supplied ligand chemistry, integer residue IDs, concrete contig lengths, and at most 26 output chains. Inputs with more than three chains require at least 32 tokens, matching the upstream interchain-neighbor reservation. Alternate location A is selected; ambiguous duplicate atoms and insertion-code selections are rejected. Large-scale design success rates, modified polymers, arbitrary SMILES chemistry, and physical relaxation are outside the verified matrix. No generated structure is silently repaired or filtered to improve benchmark geometry.

Cache modes are `auto`, `opfs`, `cache-storage`, and `none`. Storage denial falls back to verified network reads. Storage waits are abort-aware; underlying browser writes may still finish with already-verified bytes. Safetensors parsing rejects duplicate keys, malformed metadata, missing initial `{`, invalid spans and unclaimed data.

## Reproduce and package

Final verification: **61/61 `npm test` tests and 4/4 `npm run test:gpu` suites passed**, with no skips. [Recorded test results](fixtures/test-results.json).

```bash
export PATH=/home/ruh/.nvm/versions/node/v22.21.1/bin:$PATH
npm test
npm run test:gpu
npm run summarize
npm run serve
```

Open `http://127.0.0.1:8080/demo/`. The demo performs real generation in a Worker, supports monomers/motifs, progress and cancellation, previews the backbone, persists the resolved seed, and exports PDB/mmCIF.

Offline regeneration:

```bash
npm run export:weights
/home/ruh/miniconda3/envs/foundry/bin/python -B tools/verify_exports.py
npm run oracle
npm run oracle:conditioned
/home/ruh/miniconda3/envs/foundry/bin/python -B tools/export_chemistry.py
/home/ruh/miniconda3/envs/foundry/bin/python -B tools/generate_alignment_oracle.py
/home/ruh/miniconda3/envs/foundry/bin/python -B tools/generate_neighbour_oracle.py
/home/ruh/miniconda3/envs/foundry/bin/python -B tools/generate_quantized_oracle.py
/home/ruh/miniconda3/envs/foundry/bin/python -B tools/generate_sampler_oracle.py
```

The Python tools prefer pinned source and project-local AtomWorks 3.0.4/Biotite 1.6.0 dependencies in `.cache/python-deps` (see `tools/requirements-export.txt`). Tests use the supplied Chromium/GB10, Biopython, Playwright, and ProteinDesignSpace installations; unavailable required dependencies fail rather than silently skip tests. Browser profiles/caches are created in `/tmp` and removed after testing.

`node sync-upstream.mjs TARGET` previews packaging; `--write` stages every runtime module, demo, architecture, server script, README and license, then publishes by directory rename with rollback. Readers never see mixed generations. Replacing an existing directory has a brief absent-target interval; interruption can leave `TARGET.sync-lock/previous` for recovery. Weight bundles and bulky numerical fixtures are deliberately separate assets. Keep the existing `weights/` symlink when deploying this checkout.
