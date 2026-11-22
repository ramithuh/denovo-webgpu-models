# Protein model porting references

Use the [One-Shot WebGPU Model Porting Prompt](../docs/PORTING_A_MODEL.md) for the remaining Wave 2 port (`EvoDiff-Seq OA_DM_38M`, `Original RFdiffusion`, and `ESM3 Open 1.4B` have been promoted to `models/`).

| Paper and PyTorch comparison | What works here | What would produce a model result |
| --- | --- | --- |
| [BoltzGen](boltzgen/README.md) | A byte-range probe and offline replay verify the ZIP indexes of two PyTorch checkpoint archives. Full-file hashes are publisher metadata. | Upstream loads the weights and computes atom coordinates. Dynamic-length/dynamic-step WebGPU upgrade is in progress. |

Replay the four CPU and file-format checks offline with Node 22.21 or newer and Python 3. The RFdiffusion page linked above has a separate hardware smoke test.

```sh
bash tools/porting/check.sh
```

## Outstanding milestones

1. **Architecture and export.** The first outstanding deliverable for each port is its upstream mode and input coverage matrix. The standard also calls for three precision exports, upstream round-trip measurements, and deterministic intermediate traces for each supported mode.
2. **CPU reference.** Complete model replay and comparison against those traces remain open.
3. **Hardware WebGPU.** Model-level numerical parity, buffer cleanup, worker and cache behavior, and cold and warm timings remain open. Compare GPU arithmetic against an upstream reference at the same weight precision. Measure quantization error separately against the original FP32 weights.
4. **Packaging.** Standalone demos, vendor-sync smoke tests, and model-level results remain open for all four.
