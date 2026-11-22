# Pinned architecture and mode audit

Source: `microsoft/protein-frame-flow` at
`f50d8dbbdae827be291e9f73d732b61b195f8816`.
The MIT-licensed official inference config selects `weights/pdb/published.ckpt`.
Only that checkpoint is extracted. The archive also contains SCOPE and PDB
amortization checkpoints; neither is extracted or exported.

| Component | Pinned inference behavior | Port |
|---|---|---|
| Node embedding | 128 positional + 1 diffuse mask + 128 SO(3) time + 128 R3 time → 256 | Dynamic sine/cosine kernels and linear layer |
| Pair embedding | Node projections, relative positions, 22 distance bins each for current and self-conditioned translations, diffuse mask pair; MLP → 128 | Dynamic pair kernels, 3 linear layers and LayerNorm |
| IPA | 8 heads, 16 scalar channels, 8 query/key points, 12 value points; edge bias and 32 down-projected pair channels | Packed-weight projections, world-frame point transform, geometric attention, scalar/point/pair aggregation |
| Transformer | 2 post-norm encoder layers per block, width 256, 4 heads, feedforward width 256, ReLU, no inference dropout | QKV, softmax attention, output projection, residuals, norms, MLP |
| Node transition | 3 linear layers, ReLU, residual, LayerNorm | Modular kernels |
| Backbone update | 6-channel local quaternion-vector/translation update | Quaternion composition and normalization; masked updates |
| Edge transition | Node dilation 2, 384-channel pair trunk, 2 layers, residual, projection, norm | Dynamic pair concatenation and MLP |
| Trunk depth | 6; no edge transition after final block | Handwritten JS control flow; no captured graph |
| Integration | R3 Euler, SO(3) logarithm/exponential geodesic step; rate 10 exponential or linear schedule | Dedicated WGSL kernels, dynamic loop in JS |
| Reconstruction | Alanine N, Cα, C, O, Cβ; neighboring-frame oxygen correction for samples | Shared pure-JS geometry and PDB formatting |

Translations enter and leave in Å. Internal rigid translations use nm. There is
no sequence generator, confidence head, sidechain prediction, or evaluation
pipeline in this checkpoint. Atom reconstruction and serialization are shared
geometry utilities; every neural operation and the WebGPU integration step run
on the GPU. Forward geometry preserves upstream zero-psi oxygen; sampling adds
the upstream neighboring-frame oxygen correction.

## Mode matrix

| Mode | Pinned source / selected checkpoint | Package status |
|---|---|---|
| Unconditional | `inference_unconditional.yaml`, PDB checkpoint | Implemented and parity tested |
| Self-conditioning | `trans_sc`, default enabled in sampler | Implemented; network tested both with and without SC |
| Fixed-frame motif scaffolding | `Interpolant.sample` fixed-mask path runs with PDB checkpoint | Implemented and parity tested; this does **not** make PDB weights a learned amortization model |
| Learned motif amortization | `inference_scaffolding.yaml` selects **different** `pdb_amortization/published.ckpt` | Outside the one-checkpoint package |
| Gradient-guided motif scaffolding | Guidance config selects PDB, but its nesting is incorrect; forcing guidance fails because `so3_utils.sample_wrapped_normal` is absent | Explicitly rejected; reproduced failure in `fixtures/upstream_mode_probe.json` |
| Translation potential gradients | Optional upstream autograd callback, no supplied checkpoint inference preset | No arbitrary Python/autograd callback API |
| Multi-chain / multimer | Upstream README excludes multichain benchmark target 6VW1 | Unsupported |

The pinned sampler updates `batch['t']` on its final network evaluation but
leaves `so3_t` and `r3_t` at the previous grid time. The port deliberately
preserves this detail. The upstream valid sampling domain is integer `T >= 2`:
`T=1` leaves required time fields uninitialized. `L >= 1` is dynamic, subject to
available GPU memory and device buffer limits. Batching is one backbone per call.
Use separate engines/workers for concurrent requests.

## Numerical boundary

The network contains hard distogram bins. Small floating-point differences can
change an edge feature and propagate through a multi-step trajectory. The
reported `<1e-4` bound is for the committed parity matrix, not a universal bound
for every seed and T. Exploratory T=5 and T=12 failures are retained in
`fixtures/numerical_stress.json`; they are not represented as passing tests.
See the README for reproduction and precise coverage.
