# ESM3 Open sequence and structure input checks

## Paper, selected checkpoint, and browser coverage

Hayes et al. studied ESM3 at 1.4B, 7B, and 98B parameters. The laboratory-tested fluorescent protein esmGFP came from the **base 7B** model. Biohub identifies the March 2024 small, medium, and large models as the paper checkpoints. The selected Hugging Face release, **`biohub/esm3-sm-open-v1`**, is a separately named 1.4B Open model. The paper’s fluorescence measurements apply to the 7B experiment. [Published paper](https://doi.org/10.1126/science.ads0018) · [Authors’ methods, A.5.1](https://www.biorxiv.org/content/10.1101/2024.07.01.600583v1.full) · [Published model identifiers](https://github.com/Biohub/esm/blob/d49ee4606ea4a95a8bc7388171a6e21fefdf1c26/_assets/ESM3_README.md) · [Authors’ Open release](https://www.evolutionaryscale.ai/blog/esm3-release)

| Paper feature | Selected Open model in upstream Python | Evidence in this project |
| --- | --- | --- |
| Sequence representation and prediction | Tokenizer and sequence output head | Node replay of the canvas FASTA formatter matches all 76 reference token IDs on two fixed proteins. Sequence predictions are untested. |
| Three-dimensional structure representation and prediction | Coordinate inputs, learned structure encoder, structure output head and decoder | Node replay of the canvas PDB parser matches backbone coordinates and presence flags for one chain and one missing-atom case. Learned structure tokens and predicted coordinates are untested. |
| Function and auxiliary properties | Output heads for function, residue annotations, secondary structure, and solvent accessibility | Local fixtures cover sequence and backbone inputs only. These output heads and joint multimodal inference are untested. |

The Python features above come from the pinned model definition and constructor. The two replays run on CPU with the original tokenizer and PDB reader as references. Browser hardware inference, latency, memory use, and model-output parity remain unmeasured. [Python input and output definitions](https://github.com/Biohub/esm/blob/d49ee4606ea4a95a8bc7388171a6e21fefdf1c26/esm/models/esm3.py) · [Open constructor](https://github.com/Biohub/esm/blob/d49ee4606ea4a95a8bc7388171a6e21fefdf1c26/esm/pretrained.py)

## Reproduce the input checks

The canvas exports chains as colon-separated FASTA through `formatMultiChainFasta`. This CPU check uses that production formatter and compares the resulting sequence-track IDs against Biohub’s tokenizer, including the initial `<cls>`, final `<eos>`, and interior `|` chain break.

| Deposited structure | Chains used | Residues | Matching token IDs |
| --- | --- | ---: | ---: |
| [Trp-cage, 1L2Y](https://www.rcsb.org/fasta/entry/1L2Y/display) | A | 20 | 22 / 22 |
| [Insulin, 4INS](https://www.rcsb.org/fasta/entry/4INS/display) | A and B | 21 + 30 | 54 / 54 |

For 4INS the chain break is token 31 at zero-based position 22. The fixture uses one A/B pair from the deposited structure. The other deposited pair comprises chains C and D.

For 1L2Y the dashboard’s `parseStructureText` and ESM’s `ProteinChain.from_pdb` also agree on residue identity, author numbering, N/CA/C presence and observed coordinates. The fixture retains the 80 N, CA, C and O records from model 1, chain A, in deposited order. The full RCSB file and this excerpt produced the same upstream coordinate digest. [Deposited 1L2Y PDB](https://files.rcsb.org/download/1L2Y.pdb) · [Biohub structure parser](https://github.com/Biohub/esm/blob/d49ee4606ea4a95a8bc7388171a6e21fefdf1c26/esm/utils/structure/protein_chain.py)

| 1L2Y backbone input | Residue identifiers matched | N/CA/C presence flags matched | Observed coordinate components matched |
| --- | ---: | ---: | ---: |
| Model 1 as deposited | 20 / 20 | 60 / 60 | 180 / 180 |
| Same records with the N atom of A1 removed | 20 / 20 | 60 / 60 | 177 / 177 |

Coordinates are compared byte-for-byte via SHA-256 of little-endian float32 values, in residue, atom (N, CA, C), and axis (x, y, z) order. The missing-atom control compares the presence flags separately and hashes the coordinates of observed atoms only.

From the repository root, run the offline check with Node 22.21 or newer.

```sh
node tools/porting/esm3/check-input.mjs
```

To independently verify the stored IDs and coordinate digests in an environment with the upstream Python package, run the following command. The reference run used `esm==3.2.3`, `transformers==4.48.1`, and `tokenizers==0.21.4`.

```sh
HF_HUB_OFFLINE=1 python tools/porting/esm3/check_reference.py
```

The pinned Biohub source defines the vocabulary, special-token IDs, and sequence tokenization rules. Its ESM3 Open constructor specifies 48 layers, width 1536, 24 attention heads, and `v_heads=256`. These dimensions come from source inspection. Learned structure-token encoding, numerical model outputs, GPU inference, and browser performance remain unverified. [Vocabulary](https://github.com/Biohub/esm/blob/d49ee4606ea4a95a8bc7388171a6e21fefdf1c26/esm/utils/constants/esm3.py) · [Sequence tokenizer](https://github.com/Biohub/esm/blob/d49ee4606ea4a95a8bc7388171a6e21fefdf1c26/esm/tokenization/sequence_tokenizer.py) · [Model constructor](https://github.com/Biohub/esm/blob/d49ee4606ea4a95a8bc7388171a6e21fefdf1c26/esm/pretrained.py)

## Hugging Face release readiness

Biohub’s existing model card at revision `47f0545` identifies the license as MIT. The official Hugging Face API reported `private=false` and `gated=false` at that revision. The upstream repository lists the core checkpoint, structure encoder and decoder, and function decoder as PyTorch weight files. The local checks read only small input fixtures. A separate browser release still lacks exported precision bundles and manifests, a runnable WebGPU engine, and hardware parity results. [Pinned model card](https://huggingface.co/biohub/esm3-sm-open-v1/blob/47f0545b2b6daf26a93439a3cd610f4f7f3d5478/README.md) · [Upstream weight inventory](https://huggingface.co/biohub/esm3-sm-open-v1/tree/47f0545b2b6daf26a93439a3cd610f4f7f3d5478/data/weights) · [Biohub license](https://github.com/Biohub/esm/blob/d49ee4606ea4a95a8bc7388171a6e21fefdf1c26/LICENSE.md) · [Access metadata](https://huggingface.co/api/models/biohub/esm3-sm-open-v1)
