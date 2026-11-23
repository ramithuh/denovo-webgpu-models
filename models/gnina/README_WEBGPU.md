# GNINA 1.3 — Pure WebGPU 3D CNN Protein–Ligand Scorer & Pose Optimizer

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
  baseUrl: './weights/gnina/fp16',
  precision: 'fp16',
});

const result = await engine.scorePose(receptorPdbText, ligandSdfText);
// { cnnScore, cnnAffinity, cnnVS, vina, models: { dense_1.3, dense_1.3_PT_KD_3, crossdock_default2018_KD_4 } }

engine.destroy();
```

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
