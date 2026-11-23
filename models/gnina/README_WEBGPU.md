# GNINA 1.3 — WebGPU protein–ligand scoring and flexible-ligand docking

Zero-dependency, browser-native **WebGPU** implementation of **GNINA 1.3** (`gnina/gnina`) — pure-JS Smina 28-channel atom typing, Vina empirical energy decomposition, pure WGSL 3D Gaussian atom-density gridding (`48×48×48×28`, `0.5 Å` resolution), and the full 3-model default GNINA 1.3 3D CNN scoring ensemble (`dense_1.3`, `dense_1.3_PT_KD_3`, `crossdock_default2018_KD_4`).

## Provenance & License
- **Upstream Repository:** `https://github.com/gnina/gnina`
- **Pinned Commit:** `97fa6bcc465d62b40d87ccc85d3c53e9921a7cc0`
- **License:** Dual Apache-2.0 / GPLv2 (see `LICENSE`, `LICENSE.APACHE`, `LICENSE.GNU`)
- **Primary Citation:** A. McNutt, Y. Li, R. Meli, R. Aggarwal, D. R. Koes, *"GNINA 1.3: the next increment in molecular docking with deep learning"*, *J. Cheminformatics*, 2025.

## Exported Weight Bundles (`gnina-safetensors-v1`)

All 3 default GNINA 1.3 ensemble models (`1,765,065` total parameters, `178` tensors) fit inside a single shard (`< 8 MB` even in FP32):

| Precision | Total Params | Tensors | Bundle Size | GB10 Resident VRAM | GB10 Peak VRAM | After `destroy()` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **FP32** | 1,765,065 | 178 | 7.08 MB | 7.06 MB | 72.15 MB | **0 B** |
| **FP16** | 1,765,065 | 178 | 3.58 MB | 3.56 MB | 68.64 MB | **0 B** |
| **INT8** | 1,765,065 | 178 | 1.84 MB | 1.81 MB | 66.90 MB | **0 B** |

## Native vs. Decoy Discrimination & Hardware WebGPU Parity (NVIDIA GB10)

Every precision was verified in real hardware WebGPU Chromium (`allowSoftwareFallback: false`, rejecting SwiftShader/llvmpipe/WARP) against deterministic Python reference outputs across native and perturbed decoy poses of **`184L`** and **`10GS`**:

| Complex & Pose | `CNNscore` (FP32) | `CNNaffinity` (pK) | `CNN_VS` | GPU Grid `relRMS` | GPU Max Layer `relRMS` | GPU Max Output `relRMS` (FP32 / FP16 / INT8) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `184L` native | **0.900** | 4.82 | **4.34** | `1.72e-7` | `1.09e-6` | `2.31e-6` / `1.03e-6` / `2.17e-6` |
| `184L` decoy | 0.075 | 2.61 | 0.20 | `6.02e-7` | `1.65e-6` | `2.51e-5` / `5.00e-6` / `2.13e-5` |
| `10GS` native | **0.967** | 5.98 | **5.79** | `1.94e-7` | `8.43e-7` | `6.53e-7` / `9.17e-7` / `1.08e-6` |
| `10GS` decoy | 0.022 | 4.18 | 0.09 | `1.35e-6` | `1.52e-6` | `1.53e-5` / `1.77e-5` / `4.45e-5` |

### Warm GPU Latency & Lifecycle Verification (NVIDIA GB10)
- **Warm 3-model ensemble scoring:** **65.8–68.9 ms** (FP32), **64.2–66.3 ms** (FP16), **61.1–67.0 ms** (INT8)
- **Cold init (loopback):** ~246 ms (FP16), ~250 ms (INT8), ~975 ms (FP32)
- **Verified extras:**
  - Module Web Worker (`webgpu/worker.js`) + CacheStorage offline hit (`networkAttempts: 0`)
  - Safari Private Mode / blocked `caches` graceful fallback
  - External shared `GPUDevice` reuse + in-flight `destroy()` cleanup (`0` leaked buffers)
  - Multi-pose batch scoring (`scorePoses`), per-atom & per-residue CNN ablation (`maskContributions`), Vina breakdown (`vinaBreakdown`), rigid-body pose minimization (`minimizePose`), and standalone `demo/` UI verification.

## Usage

```js
import { createGninaEngine } from './webgpu/index.js';

const engine = await createGninaEngine({
  baseUrl: './weights/default',
  precision: 'fp16',
});

const result = await engine.scorePose(receptorPdbText, ligandSdfText);
// { cnnScore, cnnAffinity, cnnVS, vinaScore, models: { dense_1.3, dense_1.3_PT_KD_3, crossdock_default2018_KD_4 } }

engine.destroy();
```

## Flexible-ligand docking

`dockPose` searches a specified pocket with a rigid receptor and one connected flexible ligand. Supply PDB/mmCIF and SDF/MOL2 text, or the typed molecules returned by `parseReceptor` and `parseLigands`. The default search follows GNINA's rescore architecture. Monte Carlo and BFGS use Vina energies in JavaScript, then the engine scores the retained poses with its CNN ensemble on WebGPU. [Upstream search implementation](https://github.com/gnina/gnina/blob/97fa6bcc465d62b40d87ccc85d3c53e9921a7cc0/gninasrc/main/main.cpp)

```js
const engine = await createGninaEngine({
  baseUrl: './weights/default', precision: 'fp32', allowSoftwareFallback: false,
});
const controller = new AbortController();
try {
  const { poses, box, stats } = await engine.dockPose(receptorPdb, ligandSdf, {
    autoboxLigand: pocketReferenceSdf,
    seed: 42, exhaustiveness: 8, numModes: 9,
    cnnScoring: 'rescore', sortBy: 'cnnScore',
    signal: controller.signal,
    onProgress({ phase, completed, total }) {
      console.log(phase, completed, total);
    },
  });
  const { pose, rank, cnnScore, cnnAffinity, cnnVS, vinaScore,
          rmsdToBest, models, contributions } = poses[0];
} finally {
  engine.destroy();
}
```

| Option | Meaning and default |
| --- | --- |
| `box` | `{ center: [x,y,z], size: [sx,sy,sz] }` in Å. Alternatively provide `autoboxLigand` or receptor `pocketResidues` such as `['A/52/GLY']`. |
| `autoboxAdd` | Padding on each face, 4 Å. Automatic side lengths are clamped to 4–64 Å. |
| `seed`, `exhaustiveness` | Seed 42, eight independently randomized chains. Browser module workers run chains in parallel. `workers: 1` runs them serially. |
| `numModes`, `minRmsd` | Return up to nine poses separated by at least 1 Å heavy-atom RMSD in input atom order. `rmsdToBest` uses the same distance to rank 1. |
| `rescoreTopK` | Maximum number of clustered Vina candidates to polish and score, default `Math.max(2 * numModes, 16)`. |
| `cnnScoring` | `'rescore'` evaluates the final candidates. `'none'` ranks by exact Vina and returns null CNN values. `'refinement'` also optimizes CNN loss by coordinate search over translations, rotations and torsions with intraligand and wall restraints. This costs additional ensemble forwards. |
| `sortBy` | Descending `'cnnScore'` (default), `'cnnAffinity'` or `'cnnVS'`, or ascending `'vina'`. |
| `signal`, `onProgress` | Cancellation and progress for affinity-map construction, completed MC chains and rescored candidates. |

The search builds rigid fragments from ligand connectivity and fixes a reference frame and dihedral convention. Each chain draws a root position uniformly inside the box, a uniform 3D orientation and independent uniform torsions. Polar hydrogens follow their parent fragments. Trilinear 0.375 Å Vina grids, intraligand pairs beyond three bonds and a quadratic wall supply analytical gradients. BFGS first relaxes mutations with bounded clash energies. Promising minima undergo full-potential relaxation, RMSD clustering and off-lattice pairwise polishing. Each returned `vinaScore` and its `contributions` come from the exact pairwise `vinaBreakdown`. `stats.evaluatedCandidates` counts MC trials across chains.

### Blinded redocking on NVIDIA GB10

Chromium 153.0.8010.12 ran the default FP32 three-model ensemble with `allowSoftwareFallback: false`. The benchmark used crystal coordinates to define the padded pocket bounds and calculate RMSD. Before docking, it scrambled every input translation, orientation and torsion with seed 1907. The docking engine received the scrambled molecule and six box values and started its own chains with seed 42.

| Fixture | Rotors | Rescore K | Top-1 RMSD (Å) | Best top-3 RMSD (Å) | Top-1 CNNscore | Grid / search / rescore (s) | Wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 184L | 2 | 18 | 0.51 | 0.51 | 0.934 | 0.03 / 0.12 / 1.28 | 1.43 |
| 10GS | 13 | 18 | 1.86 | 1.86 | 0.833 | 0.21 / 1.06 / 1.45 | 2.72 |
| 10GS | 13 | 48 | 2.12 | 1.86 | 0.867 | 0.21 / 1.06 / 3.68 | 4.95 |

RMSD allows chemically equivalent atom permutations and uses the receptor coordinate frame without alignment. With fixed atom correspondence, default top-1 RMSDs were **1.59 Å** for 184L and **1.86 Å** for 10GS. Wall time covers one call with parsed inputs and a loaded engine. The engine recorded **0 B** of live GPU buffers after destruction. The saved run includes starting RMSDs, all returned ranks, exact Vina scores and the buffer counters. [Hardware measurements](fixtures/gpu_redocking.json)

## Reproducing Export & Tests

```sh
# 1. Export sharded weights (FP32/FP16/INT8) + deterministic oracles
.venv/bin/python tools/export_gnina.py
.venv/bin/python tools/dump_oracle.py

# 2. Run CPU & hardware WebGPU parity tests
npm test
```

## Proposed `denovo.run` Adapter Mapping
- **Stage:** `score` (Protein–Ligand Docking & Affinity Scorer)
- **Inputs:** `structure` (`StructureArtifact` with receptor + ligand HETATM, or separate `ligand` SDF/PDB)
- **Outputs:** `metrics` (`CNNscore`, `CNNaffinity`, `CNN_VS`, `Vina`), optional `track` (per-residue CNN contribution ablation via `maskContributions`), optional minimized `structure` (`minimizePose`).
