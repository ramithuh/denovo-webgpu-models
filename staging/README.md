# Protein model porting references

Use the [One-Shot WebGPU Model Porting Prompt](../docs/PORTING_A_MODEL.md) for all four ports. Each has early reference evidence. All four still have Milestone 1 open.

| Paper and PyTorch comparison | What works here | What would produce a model result |
| --- | --- | --- |
| [EvoDiff-Seq OA_DM_38M](evodiff/README.md) | All 31 token IDs and 64 input embedding values for a fixed example match the upstream CPU reference. | PyTorch processes those vectors through the residual network and decoder to score each possible amino acid. The local implementation ends at the embedding lookup. |
| [Original RFdiffusion](rfdiffusion/README.md) | JavaScript and Chromium WebGPU rotate supplied coordinates into cyclic and dihedral arrangements. They match PyTorch across 288 fixture components to within 2.39 × 10⁻⁷ on NVIDIA hardware. | Upstream runs a learned denoiser to produce backbone coordinates over multiple steps. The local shader implements the symmetry operation for coordinates already supplied. |
| [ESM3 Open](esm3/README.md) | Sequence IDs, backbone coordinates, and presence flags match upstream on deposited inputs, including a missing-atom case. | The Open checkpoint's learned encoders, transformer and output heads supply predictions. The local checks stop at input preparation. |
| [BoltzGen](boltzgen/README.md) | A byte-range probe and offline replay verify the ZIP indexes of two PyTorch checkpoint archives. Full-file hashes are publisher metadata. | Upstream loads the weights and computes atom coordinates. The local inspector reads storage sizes and names. |

Replay the four CPU and file-format checks offline with Node 22.21 or newer and Python 3. The RFdiffusion page linked above has a separate hardware smoke test.

```sh
bash tools/porting/check.sh
```

## Outstanding milestones

1. **Architecture and export.** The first outstanding deliverable for each port is its upstream mode and input coverage matrix. The standard also calls for three precision exports, upstream round-trip measurements, and deterministic intermediate traces for each supported mode.
2. **CPU reference.** Complete model replay and comparison against those traces remain open.
3. **Hardware WebGPU.** Model-level numerical parity, buffer cleanup, worker and cache behavior, and cold and warm timings remain open. Compare GPU arithmetic against an upstream reference at the same weight precision. Measure quantization error separately against the original FP32 weights.
4. **Packaging.** Standalone demos, vendor-sync smoke tests, and model-level results remain open for all four.
