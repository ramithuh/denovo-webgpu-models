# Source and attribution

The architecture and flow-matching semantics are ported from NVIDIA Proteina
commit `a44b407daf6a5358e43cd68907f3e3f1cbc65fdc`:

- `proteinfoundation/nn/protein_transformer.py`
- `proteinfoundation/nn/feature_factory.py`
- `proteinfoundation/nn/pair_bias_attn/pair_bias_attn.py`
- `proteinfoundation/nn/alphafold3_pytorch_utils/modules.py`
- `proteinfoundation/utils/ff_utils/idx_emb_utils.py`
- `proteinfoundation/flow_matching/r3n_fm.py`
- `proteinfoundation/proteinflow/model_trainer_base.py`

The pinned source carries NVIDIA copyright notices for 2024 and 2025 and
`SPDX-License-Identifier: LicenseRef-NvidiaProprietary`. Its top-level terms are
preserved without modification in `LICENSE.NVIDIA`; these include a
research/evaluation-only use limitation.

Component licenses are preserved in `LICENSE.OpenFold`,
`LICENSE.PairBiasAttention`, and `LICENSE.AF3Utils`. The upstream code attributes
attention to Matt McPartlon's protein-docking implementation, transition/adaptive
normalization to Lucidrains' AlphaFold3 implementation, and positional/time
embeddings to FrameFlow and diffusion implementations. OpenFold code retains
AlQuraishi Laboratory and DeepMind Apache-2.0 notices in its license.

Safetensors parsing/storage handling and the hardware Chromium test harness
follow the sibling `models/evodiff` package's architecture. No upstream repository
or checkpoint binary is part of this source package.
