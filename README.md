# denovo-webgpu-models

Zero-dependency, browser-native **WebGPU** implementations of open-weights protein design and representation models powering [denovo.run](https://denovo.run).

Weights are sharded (≤ 25 MB per `.safetensors` file) and hosted on Hugging Face under `ramithuh/*-webgpu` — this repository contains only pure ES modules, WGSL compute shaders, reproducible Python export/oracle scripts, tests, and standalone browser demos.

## Verified WebGPU Models (`models/`)

| Model | Directory | Task | Hugging Face Weights | Status |
| --- | --- | --- | --- | --- |
| **ESM-C 300M** | [`models/esmc/`](models/esmc/README_WEBGPU.md) | Protein embeddings & masked LM | [`ramithuh/esmc-webgpu`](https://huggingface.co/ramithuh/esmc-webgpu) | Verified end-to-end GPU parity |
| **ProtPardelle-1c** | [`models/protpardelle/`](models/protpardelle/README_WEBGPU.md) | All-atom backbone & motif diffusion | [`ramithuh/protpardelle-webgpu`](https://huggingface.co/ramithuh/protpardelle-webgpu) | Verified end-to-end GPU parity |
| **LigandMPNN** | [`models/ligandmpnn/`](models/ligandmpnn/README_WEBGPU.md) | Ligand/ion-aware inverse folding | [`ramithuh/ligandmpnn-webgpu`](https://huggingface.co/ramithuh/ligandmpnn-webgpu) | Verified end-to-end GPU parity |
| **LASErMPNN** | [`models/lasermpnn/`](models/lasermpnn/README_WEBGPU.md) | Sequence design + sidechain packing | [`ramithuh/lasermpnn-webgpu`](https://huggingface.co/ramithuh/lasermpnn-webgpu) | Verified end-to-end GPU parity |
| **ThermoMPNN** | [`models/thermompnn/`](models/thermompnn/README_WEBGPU.md) | Stability ΔΔG saturation scan | [`ramithuh/thermompnn-webgpu`](https://huggingface.co/ramithuh/thermompnn-webgpu) | Verified end-to-end GPU parity |
| **AntiFold / ESM-IF1** | [`models/antifold-esmif1/`](models/antifold-esmif1/README_WEBGPU.md) | Antibody CDR & inverse folding | [`ramithuh/antifold-esmif1-webgpu`](https://huggingface.co/ramithuh/antifold-esmif1-webgpu) | Verified end-to-end GPU parity |
| **Dyna-1** | [`models/dyna1/`](models/dyna1/README_WEBGPU.md) | Residue dynamics / flexibility prediction | [`ramithuh/dyna1-webgpu`](https://huggingface.co/ramithuh/dyna1-webgpu) | Verified end-to-end GPU parity |

*(Standalone flagship sibling repo: [`ramithuh/genie3-webgpu`](https://github.com/ramithuh/genie3-webgpu).)*

## Staging / In-Progress (`staging/`)

Early Milestone-0/prep notes and component checks (tokenizer, symmetry kernel, parser, archive audit) that have **not** yet reached end-to-end PyTorch parity live in [`staging/`](staging/README.md). Models are promoted from `staging/` to `models/` (and uploaded to Hugging Face) **only** after passing full end-to-end hardware WebGPU parity against the pinned PyTorch checkpoint.

## Porting a New Model

Use the canonical one-shot specification in [`docs/PORTING_A_MODEL.md`](docs/PORTING_A_MODEL.md) — **one model per port**, targeting `models/<model>/`, using existing engines like `models/esmc/` and `models/protpardelle/` as architectural templates.

## Licensing

Each model directory retains its own upstream `LICENSE` and attribution in `README_WEBGPU.md`. Upstream Python repositories are never vendored into this repo; export scripts clone pinned commits into gitignored `.cache/upstream/`.
