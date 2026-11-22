# Original RFdiffusion symmetry geometry

This directory ports the cyclic and dihedral coordinate transforms from Watson et al.'s RFdiffusion to a dependency-free JavaScript module. It rotates one fixed asymmetric unit about the origin. For Cₙ it returns n copies in order of increasing angle about z. For Dₙ it returns 2n copies, following each z rotation with the same rotated copy turned 180° about x. The input origin therefore determines the assembly center. [Pinned SymGen implementation](https://github.com/RosettaCommons/RFdiffusion/blob/86507b6538f51fce57b5a72477165f03999ed7ae/rfdiffusion/inference/symmetry.py) · [Watson et al., Nature 2023](https://doi.org/10.1038/s41586-023-06415-8)

## Paper, PyTorch sampler and WebGPU coverage

- **Paper.** Watson et al. adapted RoseTTAFold to denoise protein backbone structures iteratively. They evaluated monomer generation, symmetric assemblies, motif scaffolding and binding-protein design using structure predictions and, for selected designs, experiments. Those results describe the trained model. [Watson et al., Nature 2023](https://www.nature.com/articles/s41586-023-06415-8)
- **Pinned PyTorch sampler.** At commit `86507b6538f51fce57b5a72477165f03999ed7ae`, the driver runs a sequence of reverse-diffusion steps. Each step calls the checkpoint-backed network. When symmetry is enabled, the sampler also calls `SymGen.apply_symmetry` on the updated coordinates and sequence. Its cyclic and dihedral paths rotate the first subunit’s coordinates and copy its sequence to each subunit. [Inference driver](https://github.com/RosettaCommons/RFdiffusion/blob/86507b6538f51fce57b5a72477165f03999ed7ae/scripts/run_inference.py) · [Sampler](https://github.com/RosettaCommons/RFdiffusion/blob/86507b6538f51fce57b5a72477165f03999ed7ae/rfdiffusion/inference/model_runners.py) · [Symmetry helper](https://github.com/RosettaCommons/RFdiffusion/blob/86507b6538f51fce57b5a72477165f03999ed7ae/rfdiffusion/inference/symmetry.py)
- **Local release status.** The JavaScript and WGSL implementations cover the coordinate rotation alone. The reference fixture came from running the pinned PyTorch helper on four fixed points. On the NVIDIA GB10, the maximum error against that fixture was 2.38 × 10⁻⁷. A full RFdiffusion model release still requires a checkpoint loader, the learned network and an end-to-end sampler. Sequence replication and model-level design quality remain unverified. [Fixture capture](./capture_reference.py) · [WGSL implementation](./symmetry-webgpu.mjs) · [Browser parity check](./webgpu-smoke.html)

## Use with fixed coordinates

```js
import { expandSymmetry } from './tools/porting/rfdiffusion/symmetry.mjs'

const points = Float32Array.of(1, 2, 3, 4, 5, 6)
const copies = expandSymmetry('d3', points)
// Six copies of two points. Each consecutive six floats form one copy.
```

Supply one or more finite xyz triples for one unit, in the desired atom order. The output is a new packed Float32Array in the same coordinate units. This milestone covers fixed-coordinate geometry only. The CPU checks ran on AArch64 using Node 22.21.1 and the PyTorch CPU backend.

The WebGPU entry point compiles a compute pipeline for a caller-owned device. The host supplies rotation coefficients and the WGSL kernel rotates one point per invocation. Each call returns the packed coordinates after GPU readback.

```js
import { createSymmetryExpander } from './tools/porting/rfdiffusion/symmetry-webgpu.mjs'

const adapter = await navigator.gpu.requestAdapter()
const device = await adapter.requestDevice()
try {
  const expand = await createSymmetryExpander(device)
  const copies = await expand('d3', Float32Array.of(1, 2, 3, 4, 5, 6))
  console.log(copies)
} finally {
  device.destroy()
}
```

## Verify offline

From the repository root, run the seven reference and geometry checks with Node 22.

```sh
node tools/porting/rfdiffusion/symmetry.test.mjs
```

The fixture contains coordinates from the upstream CPU implementation for C1, D1, C3, D4 and D5. Each group starts with four off-axis points arranged into two residues. Tests compare all 288 output coordinate components in copy order with an absolute tolerance of 1e-6. The largest observed difference was 2.39e-7 or less. Separate checks verify that each copy preserves six pairwise distances and has determinant +1 for C2, C7, D3 and D8.

To regenerate the reference, use Python with PyTorch, SciPy, NumPy and pyrsistent installed. Save the pinned source file locally, then pass its path to the capture script. The script verifies the source SHA-256 before executing SymGen on CPU. The checked-in fixture used PyTorch 2.9.0, SciPy 1.16.2 and NumPy 2.3.4. [Upstream source file](https://raw.githubusercontent.com/RosettaCommons/RFdiffusion/86507b6538f51fce57b5a72477165f03999ed7ae/rfdiffusion/inference/symmetry.py)

```sh
python tools/porting/rfdiffusion/capture_reference.py /path/to/symmetry.py > /tmp/symmetry-reference.json
cmp /tmp/symmetry-reference.json tools/porting/rfdiffusion/symmetry-reference.json
```

## Run the optional browser GPU check

From the repository root, serve the fixture and smoke page on loopback.

```sh
python3 -m http.server 8147 --bind 127.0.0.1 --directory tools/porting/rfdiffusion
```

Open the [local smoke page](http://localhost:8147/webgpu-smoke.html) in Chromium with WebGPU enabled. The page prints the browser's adapter information, a pass/fail status and the maximum absolute error per group. The page requires an explicit false fallback flag and a disclosed adapter identity. It rejects known software renderer names before requesting a device. For the headless Linux run, Chromium used `--headless --enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan --disable-vulkan-surface`.

Chrome for Testing 153.0.8010.12 reported vendor `nvidia` and `isFallbackAdapter=false`. Chromium's GPU diagnostics identified NVIDIA GB10 using Vulkan 1.4.312, driver 580.159.3.0. The five captured cases passed on GPU across all 288 coordinate components. An additional D5 check repeated the unit 19 times, yielding 760 points across 12 workgroups and 2,280 coordinate components. The largest absolute difference from the pinned PyTorch output was 2.384185791015625e-7 in both checks.

To reproduce the software control, use a separate Chromium profile with the same headless options, replacing `--use-angle=vulkan` with `--use-angle=swiftshader --use-vulkan=swiftshader --enable-unsafe-swiftshader`. On this host the browser reported vendor `google`, architecture `swiftshader` and `isFallbackAdapter=true`. The page returned `fail` with zero cases executed. Simulated browser controls for a missing adapter, blank identity fields, placeholder-only identity fields, a missing fallback flag and an llvmpipe identity paired with a false fallback flag also failed before dispatch.
